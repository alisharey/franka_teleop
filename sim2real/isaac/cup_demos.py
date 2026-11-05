"""Scripted cup-stacking demos in Isaac Lab with physics, many environments in parallel, recorded like our
real episodes.

Scene from sim2real/scene (table + texture, cup sizes/colors, calibrated cameras): hollow rigid-body cups
that nest, the Franka Panda (same kinematics as the FR3) starting at the saved home pose. Each episode
places both cups anywhere in a reachable table area (--region), then a scripted hand (above the picked
cup, down, close, lift, above the other cup, down, open, up; min-jerk) is followed through our teleop's
tracking law (TeleopTracking: twist = 5 x error, capped at 0.25 m/s / 0.4 rad/s, Cartesian S-curve) and
differential IK; the fingers really close on the cup, so a grasp can fail (failed episodes are dropped).

--num-envs scenes run side by side (one TiledCamera per camera for all of them); each environment
starts its next episode as soon as one ends, until --episodes successful episodes are saved.

Recorded at 30 Hz in the real export format (tools/baku/export_episodes.py), depth like the recorder:
    <out>/<episode_id>/cam_1_color.mp4, cam_2_color.mp4, cam_1_depth.mkv, cam_2_depth.mkv (FFV1 16-bit mm),
                       lowdim.npz, episode.json (written last: marks a complete episode)
The action is our TeleMoMa convention: [target - current position, scipy "xyz" Euler of
R_target R_current^T, gripper 1 = open], target = the scripted hand.
"""
import argparse
import json
import math
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', required=True, help='folder with scene.json, calibration.json, table_texture.png, cup_colors.json')
parser.add_argument('--robot-usd', required=True, help='local panda_instanceable.usd (mirror_usd.py)')
parser.add_argument('--episodes', type=int, default=200, help='successful episodes to save')
parser.add_argument('--num-envs', type=int, default=32)
parser.add_argument('--env-spacing', type=float, default=6.0)
parser.add_argument('--picked', default='blue')
parser.add_argument('--target', default='red')
parser.add_argument('--task', default='stack the blue cup on top of the red one')
parser.add_argument('--home', default='-0.00633,-0.99146,0.02246,-2.68242,0.05889,1.69572,0.88545')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--region', default='0.30,0.65,-0.28,0.32',
                    help='x0,x1,y0,y1 [m]: both cups are sampled anywhere in it (reachable, inside the teleop workspace)')
parser.add_argument('--min-separation', type=float, default=0.14)
parser.add_argument('--hand-speed', default='0.08,0.16', help='scripted hand mean speed range per episode [m/s] '
                    '(min-jerk peak = 1.9x; the arm is capped at 0.25)')
parser.add_argument('--policy', default='', help='evaluate a policy instead of scripting: HOST:PORT of '
                    'sim2real/policy_server.py; --episodes random episodes over --num-envs, results in <out>/eval.json')
parser.add_argument('--eval-seconds', type=float, default=45.0, help='time limit per evaluation episode')
parser.add_argument('--eval-videos', type=int, default=10, help='save the policy input video of the first n episodes')
parser.add_argument('--sac', action='store_true', help='train SAC with a demonstration buffer (sac_cups.py) in <out>')
parser.add_argument('--sac-init', default='', help='continue SAC from an agent checkpoint (<out>/sac/checkpoints/*.pt)')
parser.add_argument('--sac-eval', default='', help='evaluate a SAC policy (policy_*.pt from --sac)')
parser.add_argument('--sac-steps', type=int, default=4000, help='SAC environment steps (each: all environments, 10 Hz)')
parser.add_argument('--sac-demos', type=int, default=200, help='scripted demonstrations in the demo buffer')
parser.add_argument('--sac-bc-steps', type=int, default=3000, help='behavior cloning steps before RL')
parser.add_argument('--sac-bc-weight', type=float, default=1.0)
parser.add_argument('--sac-memory', type=int, default=2500, help='online replay steps per environment')
parser.add_argument('--sac-demo-memory', type=int, default=40000, help='demonstration transitions')
parser.add_argument('--sac-gradient-steps', type=int, default=16, help='updates per environment step')
parser.add_argument('--sac-eval-episodes', type=int, default=50)
parser.add_argument('--out', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

HAND_TO_EE = np.eye(4)
HAND_TO_EE[2, 3] = 0.1034  # libfranka's EE frame with the Franka Hand = panda_hand + 10.34 cm along z
PHYSICS_DT, DECIMATION = 1 / 120, 4  # control, cameras and recording at 30 Hz
W, H = 640, 480
rng = np.random.default_rng(args.seed)


# ------------------------------------------------------------------ math (batched over environments)
def skew(v):
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1), np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rotmat(v):
    """Rotation vectors (..., 3) -> matrices (..., 3, 3)."""
    angle = np.linalg.norm(v, axis=-1)[..., None, None]
    k = skew(v / np.maximum(angle[..., 0], 1e-12))
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def rotvec(r):
    """Matrices (..., 3, 3) -> rotation vectors (..., 3) (angles < pi)."""
    angle = np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    v = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
    s = np.sin(angle)
    scale = np.where(s > 1e-6, angle / (2 * np.maximum(s, 1e-6)), 0.5)
    return v * scale[..., None]


def quat_from_matrix(r):
    """(..., 3, 3) -> (..., 4) w, x, y, z."""
    w = np.sqrt(np.maximum(0, 1 + r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2])) / 2
    x = np.copysign(np.sqrt(np.maximum(0, 1 + r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2])) / 2, r[..., 2, 1] - r[..., 1, 2])
    y = np.copysign(np.sqrt(np.maximum(0, 1 - r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2])) / 2, r[..., 0, 2] - r[..., 2, 0])
    z = np.copysign(np.sqrt(np.maximum(0, 1 - r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2])) / 2, r[..., 1, 0] - r[..., 0, 1])
    q = np.stack([w, x, y, z], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def matrix_from_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def euler_xyz(r):
    """scipy 'xyz' (extrinsic): R = Rz(c) Ry(b) Rx(a) -> (a, b, c), as TeleMoMa's action."""
    return np.stack([np.arctan2(r[..., 2, 1], r[..., 2, 2]), np.arcsin(np.clip(-r[..., 2, 0], -1, 1)),
                     np.arctan2(r[..., 1, 0], r[..., 0, 0])], -1)


def min_jerk(a, b, s):
    s = np.clip(s, 0, 1)
    return a + (b - a) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


# ------------------------------------------------------------------ teleop controller
class TeleopTracking:
    """Our teleop's tracking law (src/gamepad_teleop.cpp, config/teleop.conf) for n arms: twist = gain x (hand
    target - actual end-effector pose), capped at the max speeds, then the Cartesian S-curve (shape_group at
    1 kHz, 80 % jerk margin). The shaped twist is integrated into the pose the IK follows."""

    def __init__(self, n, gain=5.0, max_linear=0.25, max_angular=0.40, linear_acc=1.0, linear_jerk=10.0,
                 angular_acc=2.5, angular_jerk=25.0):
        self.gain, self.max_linear, self.max_angular = gain, max_linear, max_angular
        self.limits = ((linear_acc, linear_jerk), (angular_acc, angular_jerk))
        self.velocity = np.zeros((n, 6))
        self.acceleration = np.zeros((n, 6))
        self.command = np.tile(np.eye(4), (n, 1, 1))

    def reset(self, ids, poses):
        self.velocity[ids] = 0
        self.acceleration[ids] = 0
        self.command[ids] = poses

    def twist(self, target, actual):
        out = np.zeros((len(target), 6))
        out[:, :3] = self.gain * np.clip(target[:, :3, 3] - actual[:, :3, 3], -1, 1)
        out[:, 3:] = self.gain * rotvec(target[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1))
        for first, cap in ((0, self.max_linear), (3, self.max_angular)):
            norm = np.linalg.norm(out[:, first:first + 3], axis=1, keepdims=True)
            out[:, first:first + 3] *= np.minimum(1.0, cap / np.maximum(norm, 1e-12))
        return out

    def _shape(self, target, first, max_acc, max_jerk, dt=1e-3, margin=0.8):
        g = slice(first, first + 3)
        error = target[:, g] - self.velocity[:, g]
        acc = self.acceleration[:, g]
        e = np.linalg.norm(error, axis=1, keepdims=True)
        a = np.linalg.norm(acc, axis=1, keepdims=True)
        snap = (e < 5.0 * max_jerk * dt * dt) & (a < max_jerk * dt)
        desired = error / np.maximum(e, 1e-12) * np.minimum(max_acc, np.sqrt(2.0 * margin * max_jerk * e))
        jerk = (desired - acc) / dt
        jerk *= np.minimum(1.0, max_jerk / np.maximum(np.linalg.norm(jerk, axis=1, keepdims=True), 1e-12))
        acc = acc + jerk * dt
        acc *= np.minimum(1.0, max_acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12))
        self.velocity[:, g] = np.where(snap, target[:, g], self.velocity[:, g] + acc * dt)
        self.acceleration[:, g] = np.where(snap, 0.0, acc)

    def step(self, target, actual, dt):
        """Advances dt seconds (1 ms shaper ticks); returns the poses for the IK."""
        desired = self.twist(target, actual)
        for _ in range(max(1, int(round(dt / 1e-3)))):
            self._shape(desired, 0, *self.limits[0])
            self._shape(desired, 3, *self.limits[1])
            self.command[:, :3, 3] += self.velocity[:, :3] * 1e-3
            self.command[:, :3, :3] = rotmat(self.velocity[:, 3:] * 1e-3) @ self.command[:, :3, :3]
        # stay anchored to the real arm (the robot executes velocities on its own state)
        drift = self.command[:, :3, 3] - actual[:, :3, 3]
        norm = np.linalg.norm(drift, axis=1, keepdims=True)
        self.command[:, :3, 3] = actual[:, :3, 3] + drift * np.minimum(1.0, 0.08 / np.maximum(norm, 1e-12))
        return self.command


# ------------------------------------------------------------------ scene
def texture_material(stage, path, texture_file):
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path + '/surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7)
    reader = UsdShade.Shader.Define(stage, path + '/uv')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    texture = UsdShade.Shader.Define(stage, path + '/texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture_file))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    return material


def add_textured_top(stage, path, x0, x1, y0, y1, material, z=0.0005):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z), Gf.Vec3f(x1, y1, z), Gf.Vec3f(x0, y1, z)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    mesh.CreateSubdivisionSchemeAttr('none')
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def add_cup(stage, path, bottom_d, top_d, height, look, physics_material, mass, wall=0.003, segments=24):
    """Hollow cup standing upside down: open wide rim on the table, closed narrow end on top, so a cup
    lowered over another slides down over it. Visual: outer + inner wall + top cap. Collision: a ring of
    thin wall boxes along the taper plus a cap disk (PhysX cannot collide with a hollow mesh directly)."""
    rb, rt = bottom_d / 2, top_d / 2
    UsdGeom.Xform.Define(stage, path)
    n = 48
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = [(rb, 0.0), (rt, height), (rt - wall, height - wall), (rb - wall, 0.0)]  # outer bottom/top, inner top/bottom
    points = [(r * np.cos(a), r * np.sin(a), z) for r, z in rings for a in angles] + [(0, 0, height), (0, 0, height - wall)]
    counts, indices = [], []
    for ring in range(3):  # outer wall, top lip, inner wall
        for i in range(n):
            j = (i + 1) % n
            counts.append(4)
            indices += [ring * n + i, ring * n + j, (ring + 1) * n + j, (ring + 1) * n + i]
    for i in range(n):  # rim (bottom annulus) and the two caps
        j = (i + 1) % n
        counts += [4, 3, 3]
        indices += [3 * n + i, 3 * n + j, j, i, 4 * n, n + i, n + j, 4 * n + 1, 2 * n + j, 2 * n + i]
    mesh = UsdGeom.Mesh.Define(stage, path + '/visual')
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    sim_utils.bind_visual_material(path + '/visual', look)
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    slant = math.atan2(rb - rt, height)
    length = math.hypot(height, rb - rt)
    r_mid = (rb + rt) / 2 - wall / 2 * math.cos(slant)
    width = 2 * r_mid * math.sin(math.pi / segments) * 1.15
    tilt = Gf.Rotation(Gf.Vec3d(0, 1, 0), -math.degrees(slant))
    for k in range(segments):
        theta = 2 * math.pi * k / segments
        box = UsdGeom.Cube.Define(stage, '%s/wall_%02d' % (path, k))
        box.CreateSizeAttr(1.0)
        rotation = tilt * Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(theta))
        xform = UsdGeom.Xformable(box)
        xform.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(theta), r_mid * math.sin(theta), height / 2))
        xform.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
        xform.AddScaleOp().Set(Gf.Vec3f(wall, width, length))
        box.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        sim_utils.bind_physics_material(box.GetPath().pathString, physics_material)
    cap = UsdGeom.Cylinder.Define(stage, path + '/collider_cap')
    cap.CreateRadiusAttr(rt)
    cap.CreateHeightAttr(wall)
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(0, 0, height - wall / 2))
    cap.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    sim_utils.bind_physics_material(path + '/collider_cap', physics_material)


def build_scene(scene, origins):
    """Lights, floor and shared materials once; table and cups per environment under /World/envs/env_<i>."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    data = Path(args.data)
    table = scene['table']
    table_height = 0.75
    extent = float(np.abs(origins).max()) * 2 + 10
    floor = sim_utils.CuboidCfg(size=(extent, extent, 0.02), collision_props=sim_utils.CollisionPropertiesCfg(),
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.37)))
    floor.func('/World/floor', floor, translation=(0, 0, -table_height - 0.01))
    sim_utils.DomeLightCfg(intensity=700.0).func('/World/dome', sim_utils.DomeLightCfg(intensity=700.0))
    sun = sim_utils.DistantLightCfg(intensity=600.0, angle=3.0)
    sun.func('/World/sun', sun, orientation=(0.9, 0.2, 0.3, 0.0))
    # shared materials: the cups' colors as matched to the real cups (cup_colors.json, linear RGB)
    colors_file = data / 'cup_colors.json'
    colors = json.loads(colors_file.read_text())['final_diffuse_linear'] if colors_file.exists() else {}
    sizes = {}
    for color in (args.picked, args.target):
        s = scene['objects'][color]
        sizes[color] = (s['bottom_diameter_m']['median'], s['top_diameter_m']['median'], s['height_m']['median'])
        if color not in colors:
            hsv = np.median([o['hsv'] for e in scene['per_episode'] for o in e['objects'] if o['color'] == color], axis=0)
            bgr = cv2.cvtColor(np.uint8([[hsv.astype(np.uint8)]]), cv2.COLOR_HSV2BGR)[0, 0] / 255.0
            colors[color] = [float(c) ** 2.2 for c in bgr[::-1]]
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(colors[color]), roughness=0.4)
        look.func('/World/Looks/cup_%s' % color, look)
    cup_physics = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8)
    cup_physics.func('/World/Looks/cup_physics', cup_physics)
    texture = data / 'table_texture.png'
    table_texture = texture_material(stage, '/World/Looks/table_texture', texture.resolve()) if texture.exists() else None
    lx, ly = table['x_m'][1] - table['x_m'][0], table['y_m'][1] - table['y_m'][0]
    cx, cy = (table['x_m'][1] + table['x_m'][0]) / 2, (table['y_m'][1] + table['y_m'][0]) / 2
    top = sim_utils.CuboidCfg(size=(lx, ly, 0.03), collision_props=sim_utils.CollisionPropertiesCfg(),
                              physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.86, 0.84)))
    for i, origin in enumerate(origins):
        root = '/World/envs/env_%d' % i
        UsdGeom.Xformable(UsdGeom.Xform.Define(stage, root)).AddTranslateOp().Set(Gf.Vec3d(*origin.tolist()))
        top.func(root + '/table/top', top, translation=(cx, cy, -0.015))
        if table_texture:
            add_textured_top(stage, root + '/table/textured_top', table['x_m'][0], table['x_m'][1], table['y_m'][0],
                             table['y_m'][1], table_texture)
        for color in (args.picked, args.target):
            add_cup(stage, '%s/cup_%s' % (root, color), *sizes[color], '/World/Looks/cup_%s' % color,
                    '/World/Looks/cup_physics', mass=0.02)
    return sizes


# ------------------------------------------------------------------ recording
class EpisodeWriter:
    """Streams one episode's camera frames to ffmpeg (color mp4, depth FFV1 16-bit like the recorder)."""

    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        base = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-s', '%dx%d' % (W, H), '-r', '30']
        self.procs = {}
        for cam in ('cam_1', 'cam_2'):
            self.procs[cam + '_color'] = subprocess.Popen(
                base + ['-pix_fmt', 'rgb24', '-i', '-', '-c:v', 'mpeg4', '-q:v', '2', str(folder / ('%s_color.mp4' % cam))],
                stdin=subprocess.PIPE)
            self.procs[cam + '_depth'] = subprocess.Popen(
                base + ['-pix_fmt', 'gray16le', '-i', '-', '-c:v', 'ffv1', str(folder / ('%s_depth.mkv' % cam))],
                stdin=subprocess.PIPE)
        self.log = {k: [] for k in ('ee', 'q', 'width', 'target', 'open')}

    def close(self):
        for proc in self.procs.values():
            proc.stdin.close()

    def done(self):
        return all(proc.poll() is not None for proc in self.procs.values())


def save_lowdim(folder, episode_id, log, positions):
    n = len(log['ee'])
    ee = np.array(log['ee'])
    target = np.array(log['target'])
    action = np.zeros((n, 7))
    action[:, :3] = np.clip(target[:, :3, 3] - ee[:, :3, 3], -1, 1)
    action[:, 3:6] = np.clip(euler_xyz(target[:, :3, :3] @ ee[:, :3, :3].transpose(0, 2, 1)), -1, 1)
    action[:, 6] = log['open']
    delta = np.zeros((n, 6))
    delta[:-1, :3] = ee[1:, :3, 3] - ee[:-1, :3, 3]
    delta[:-1, 3:] = rotvec(ee[1:, :3, :3] @ ee[:-1, :3, :3].transpose(0, 2, 1))
    quats = quat_from_matrix(ee[:, :3, :3])
    np.savez_compressed(folder / 'lowdim.npz', t=np.arange(n) / 30.0, stamp_ns=(np.arange(n) * 1e9 / 30).astype(np.int64),
                        ee_pos=ee[:, :3, 3], ee_quat=quats[:, [1, 2, 3, 0]], ee_rotvec=rotvec(ee[:, :3, :3]),
                        q=np.array(log['q']), gripper_width=np.array(log['width']), gripper_open=np.array(log['open']),
                        telemoma_action=action, movement_enabled=np.ones(n, bool), action=action, delta_pose=delta,
                        frame_cam_1=np.arange(n), frame_cam_2=np.arange(n),
                        frame_skew_ms_cam_1=np.zeros(n), frame_skew_ms_cam_2=np.zeros(n))
    info = {'format': 'franka_teleop_baku_export/1', 'episode_id': episode_id, 'name': episode_id, 'task': args.task,
            'success': True, 'notes': 'Isaac Lab scripted demo (physics), cups at %s' % {c: v.tolist() for c, v in positions.items()},
            'source': 'isaac_lab_scripted', 'cameras': {c: {'file': '%s_color.mp4' % c, 'width': W, 'height': H,
                                                            'depth_file': '%s_depth.mkv' % c} for c in ('cam_1', 'cam_2')},
            'observation_camera': 'cam_1', 'frames': n, 'duration_s': n / 30.0, 'fps': 30.0, 'driven_fraction': 1.0,
            'action_convention': 'telemoma: [dx dy dz] = target - current [m]; [droll dpitch dyaw] = scipy xyz Euler of '
                                 'R_target R_current^T [rad]; base frame; gripper 1 = open, 0 = closed'}
    (folder / 'episode.json').write_text(json.dumps(info, indent=1))  # last: marks the episode complete


# ------------------------------------------------------------------ policy evaluation
def euler_xyz_matrix(e):
    """Inverse of euler_xyz: R = Rz(c) Ry(b) Rx(a)."""
    a, b, c = e
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def evaluate(reset_envs, place_cups, judge, control, ee_poses, home_ee, static_cam, wrist_cam, tracker, data, calib, out):
    """Runs the policy as vla/baku/deploy.py + our teleop do on the robot, in every environment at once: at the
    policy rate the background-filtered camera frames (sim2real/observation.py) go to the policy; its TeleMoMa
    action becomes a target fixed at (pose now + action), tracked by the teleop law until the next action;
    gripper with the teleop's hysteresis (open >= 0.65, close <= 0.35). An episode succeeds when the cups are
    nested and the gripper has released for 1 s; it fails at --eval-seconds."""
    import io
    import sys
    from multiprocessing.connection import Client
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sim2real.observation import BackgroundFilter
    bg = BackgroundFilter(data / 'scene.json', data / 'calibration.json', sim=True)
    host, port = args.policy.rsplit(':', 1)
    connection = Client((host, int(port)))

    def ask(**arrays):
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        connection.send_bytes(buffer.getvalue())
        return dict(np.load(io.BytesIO(connection.recv_bytes())))

    n = args.num_envs
    rate = float(ask(rate=np.ones(1))['rate_hz'])
    per_tick = max(1, int(round(30 / rate)))
    pool = ThreadPoolExecutor(16)
    results, started, envs = [], 0, [None] * n

    def begin(ids):
        nonlocal started
        ids = list(ids)[:max(0, args.episodes - started)]  # environments past the last episode stay idle at home
        if not ids:
            return
        positions = [place_cups() for _ in ids]
        reset_envs(ids, positions)
        for i, p in zip(ids, positions):
            envs[i] = {'episode': started, 'positions': p, 'tick': 0, 'warmup': 4, 'open': True, 'nested': 0,
                       'reset': True, 'video': [] if started < args.eval_videos else None}
            started += 1

    def finish(i, success):
        e = envs[i]
        results.append({'episode': e['episode'], 'success': success, 'seconds': e['tick'] / rate,
                        'positions': {c: v.tolist() for c, v in e['positions'].items()}})
        print('eval episode %d (env %d): %s after %.1f s; %d / %d done, %d successes' % (
            e['episode'], i, 'SUCCESS' if success else 'failed', e['tick'] / rate, len(results), args.episodes,
            sum(r['success'] for r in results)), flush=True)
        if e['video'] is not None:
            writer = cv2.VideoWriter(str(out / ('eval_%03d_%s.mp4' % (e['episode'], 'success' if success else 'fail'))),
                                     cv2.VideoWriter_fourcc(*'mp4v'), rate, (2 * W, H))
            for image in e['video']:
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            writer.release()
        envs[i] = None
        (out / 'eval.json').write_text(json.dumps({
            'policy': args.policy, 'episodes': len(results), 'success_rate': float(np.mean([r['success'] for r in results])),
            'region': args.region, 'seed': args.seed, 'eval_seconds': args.eval_seconds,
            'results': sorted(results, key=lambda r: r['episode'])}, indent=1))

    begin(range(n))
    while len(results) < args.episodes:
        actual = ee_poses()
        targets = np.tile(home_ee, (n, 1, 1))
        gripper = np.ones(n, bool)
        active = [i for i, e in enumerate(envs) if e is not None and e['warmup'] == 0]
        for i, e in enumerate(envs):
            if e is not None and e['warmup'] > 0:  # just reset: hold at home
                e['warmup'] -= 1
                tracker.reset([i], actual[i:i + 1])
        if active:
            frames = {}
            for key, name, cam in (('pixels', 'cam_2', static_cam), ('pixels_egocentric', 'cam_1', wrist_cam)):
                rgb = cam.data.output['rgb'][..., :3].contiguous().cpu().numpy()
                depth = torch.nan_to_num(cam.data.output['distance_to_image_plane'][..., 0] * 1000, posinf=0)
                depth = depth.clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
                frames[key] = np.stack(list(pool.map(lambda i: bg.apply(name, rgb[i], depth[i], actual[i]), active)))
            reply = ask(env=np.array(active), reset=np.array([envs[i]['reset'] for i in active]), **frames)
            for row, i in enumerate(active):
                e, action = envs[i], reply['action'][row]
                e['reset'] = False
                e['target'] = np.eye(4)
                e['target'][:3, 3] = actual[i, :3, 3] + action[:3]
                e['target'][:3, :3] = euler_xyz_matrix(action[3:6]) @ actual[i, :3, :3]
                e['open'] = True if action[6] >= 0.65 else False if action[6] <= 0.35 else e['open']
                if e['video'] is not None:
                    e['video'].append(np.hstack([frames['pixels'][row], frames['pixels_egocentric'][row]]))
        for i, e in enumerate(envs):
            if e is not None and 'target' in e and e['warmup'] == 0:
                targets[i], gripper[i] = e['target'], e['open']
        for _ in range(per_tick):
            control(targets, gripper)
        ended = []
        for i in active:
            e = envs[i]
            e['tick'] += 1
            e['nested'] = e['nested'] + 1 if judge(i) and e['open'] else 0
            if e['nested'] >= rate:  # nested and released for 1 s
                finish(i, True)
                ended.append(i)
            elif e['tick'] >= args.eval_seconds * rate:
                finish(i, False)
                ended.append(i)
        begin(ended)
    print('ISAAC_EVAL_DONE success %d / %d' % (sum(r['success'] for r in results), len(results)), flush=True)
    connection.close()


# ------------------------------------------------------------------ main
def main():
    data = Path(args.data)
    scene = json.loads((data / 'scene.json').read_text())
    calib = json.loads((data / 'calibration.json').read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_envs
    cols = int(math.ceil(math.sqrt(n)))
    origins = np.array([[(i % cols) * args.env_spacing, (i // cols) * args.env_spacing, 0.0] for i in range(n)])

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    sizes = build_scene(scene, origins)
    cups = {c: RigidObject(RigidObjectCfg(prim_path='/World/envs/env_.*/cup_%s' % c, spawn=None))
            for c in (args.picked, args.target)}
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path='/World/envs/env_.*/robot')
    robot_cfg.spawn.usd_path = args.robot_usd
    home = [float(v) for v in args.home.split(',')]
    robot_cfg.init_state.joint_pos = {'panda_joint%d' % (i + 1): home[i] for i in range(7)}
    robot_cfg.init_state.joint_pos.update({'panda_finger_joint.*': 0.04})
    # the real Franka Hand: 70 N continuous grasp force (Isaac's default 200 N squeezes fingers into the thin cup walls)
    robot_cfg.actuators['panda_hand'].effort_limit_sim = 70.0
    robot = Articulation(robot_cfg)

    def camera(path, k, pose):
        return TiledCamera(TiledCameraCfg(
            prim_path=path, update_period=0, height=H, width=W, data_types=['rgb', 'distance_to_image_plane'],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=k.flatten().tolist(), width=W,
                                                                   height=H, clipping_range=(0.01, 20.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(pose[:3, 3]), rot=tuple(quat_from_matrix(pose[:3, :3])), convention='ros')))
    static_cam = camera('/World/envs/env_.*/static_cam', np.array(calib['intrinsics']['cam_2']['K']),
                        np.array(scene['static_camera']['T_base_cam_leveled']))
    wrist_cam = camera('/World/envs/env_.*/robot/panda_hand/wrist_cam', np.array(calib['intrinsics']['cam_1']['K']),
                       HAND_TO_EE @ np.array(calib['T_ee_cam_cam_1']))
    sim.reset()

    dev = sim.device
    hand = robot.find_bodies('panda_hand')[0][0]
    arm = robot.find_joints('panda_joint.*')[0]
    fingers = robot.find_joints('panda_finger_joint.*')[0]
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose', use_relative_mode=False, ik_method='dls'),
                                  num_envs=n, device=dev)
    origins_t = torch.tensor(origins, dtype=torch.float32, device=dev)
    hand_offset = -HAND_TO_EE[:3, 3]
    tracker = TeleopTracking(n)

    def ee_poses():  # (n, 4, 4) in each environment's robot base frame
        pose = np.tile(np.eye(4), (n, 1, 1))
        pose[:, :3, 3] = (robot.data.body_pos_w[:, hand] - origins_t).cpu().numpy()
        pose[:, :3, :3] = matrix_from_quat(robot.data.body_quat_w[:, hand].cpu().numpy())
        return pose @ HAND_TO_EE

    def control(targets, gripper_open):
        """One 30 Hz control step for all environments: hand targets through the tracking law, IK follows."""
        robot.set_joint_position_target(torch.tensor(np.where(gripper_open, 0.04, 0.0)[:, None].repeat(len(fingers), 1),
                                                     dtype=torch.float32, device=dev), joint_ids=fingers)
        for _ in range(DECIMATION):
            actual = ee_poses()
            command = tracker.step(targets, actual, PHYSICS_DT) @ np.linalg.inv(HAND_TO_EE)
            pose_w = np.concatenate([command[:, :3, 3] + origins, quat_from_matrix(command[:, :3, :3])], 1)
            ik.set_command(torch.tensor(pose_w, dtype=torch.float32, device=dev))
            jacobian = robot.root_physx_view.get_jacobians()[:, hand - 1, :, arm]
            q_arm = ik.compute(robot.data.body_pos_w[:, hand], robot.data.body_quat_w[:, hand], jacobian,
                               robot.data.joint_pos[:, arm])
            robot.set_joint_position_target(q_arm, joint_ids=arm)
            # velocity feed-forward (the real arm executes the commanded velocity; a pure PD target lags ~0.2 s)
            twist = tracker.velocity.copy()
            twist[:, :3] += np.cross(twist[:, 3:], actual[:, :3, :3] @ hand_offset)  # TCP velocity -> panda_hand origin
            twist_t = torch.tensor(twist, dtype=torch.float32, device=dev)
            jj = jacobian @ jacobian.transpose(1, 2) + 0.05 ** 2 * torch.eye(6, device=dev)
            dq = (jacobian.transpose(1, 2) @ torch.linalg.solve(jj, twist_t[..., None]))[..., 0]
            robot.set_joint_velocity_target(dq, joint_ids=arm)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(PHYSICS_DT)
            for cup in cups.values():
                cup.update(PHYSICS_DT)
        sim.render()
        static_cam.update(PHYSICS_DT * DECIMATION)
        wrist_cam.update(PHYSICS_DT * DECIMATION)

    # settle once at home; the home end-effector pose is the same in every environment
    q_home = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q_home, torch.zeros_like(q_home))
    robot.set_joint_position_target(q_home)
    for _ in range(10):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT)
    home_ee = ee_poses()[0]
    tracker.reset(np.arange(n), ee_poses())

    def place_cups():
        x0, x1, y0, y1 = [float(v) for v in args.region.split(',')]
        while True:
            pos = {c: np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)]) for c in cups}
            if np.linalg.norm(pos[args.picked] - pos[args.target]) > args.min_separation and \
                    all(np.linalg.norm(p) > 0.28 for p in pos.values()):
                return pos

    def plan(positions):
        """The scripted hand as segments: ('move', poses, gripper), ('wait', pose, gripper), ('hold', pose, gripper, steps)."""
        hb, hr = sizes[args.picked][2], sizes[args.target][2]
        jitter = lambda s: rng.normal(scale=s, size=3)  # noqa: E731
        p, t = np.r_[positions[args.picked], 0.0], np.r_[positions[args.target], 0.0]
        grasp_z = hb - rng.uniform(0.02, 0.03)
        carry_z = hr + hb + rng.uniform(0.05, 0.08)
        # release when the picked cup's open rim is a little above the other cup's closed top, centred over it:
        # it then slides down over it (nests). grasp_z is the TCP height above the picked cup's rim.
        place_z = hr + rng.uniform(0.005, 0.015) + grasp_z
        points = [  # (position, gripper open, hold seconds)
            (np.r_[p[:2], hb + rng.uniform(0.08, 0.12)] + jitter(0.01) * [1, 1, 0.3], True, 0.0),
            (np.r_[p[:2], grasp_z] + jitter(0.003) * [1, 1, 0], True, 0.2),
            (np.r_[p[:2], grasp_z], False, 0.6),
            (np.r_[p[:2], carry_z], False, 0.0),
            (np.r_[t[:2], carry_z] + jitter(0.01) * [1, 1, 0.3], False, 0.0),
            (np.r_[t[:2], place_z] + jitter(0.002) * [1, 1, 0], False, 0.3),
            (np.r_[t[:2], place_z], True, 0.5),
            (np.r_[t[:2], carry_z + 0.05], True, 0.5),
        ]
        speed = rng.uniform(*[float(v) for v in args.hand_speed.split(',')])
        segments, here, opened_now = [], home_ee[:3, 3], True
        for position, opened, hold in points:
            steps = max(int(np.linalg.norm(position - here) / speed * 30), 1)
            poses = np.tile(home_ee, (steps, 1, 1))
            poses[:, :3, 3] = min_jerk(here, position, (np.arange(steps)[:, None] + 1) / steps)
            segments.append(('move', poses, opened_now))
            pose = home_ee.copy()
            pose[:3, 3] = position
            if hold > 0:  # before gripping / releasing the operator waits until the arm has arrived
                segments.append(('wait', pose, opened_now))
                segments.append(('hold', pose, opened, int(hold * 30)))
            here, opened_now = position, opened
        return segments

    def judge(i):
        """Nested: centred on the other cup, upright, and slid down well below resting on its top."""
        picked = (cups[args.picked].data.root_pos_w[i] - origins_t[i]).cpu().numpy()
        base = (cups[args.target].data.root_pos_w[i] - origins_t[i]).cpu().numpy()
        upright = matrix_from_quat(cups[args.picked].data.root_quat_w[i].cpu().numpy())[2, 2] > 0.9
        rim_height = picked[2] - base[2]
        offset = np.linalg.norm(picked[:2] - base[:2])
        judge.last = 'cup offset %.0f mm, rim %.0f mm above the other rim, %s' % (
            offset * 1000, rim_height * 1000, 'upright' if upright else 'tipped')
        return bool(offset < 0.02 and upright and rim_height < 0.6 * sizes[args.target][2])

    def reset_envs(ids, positions):
        """Arms of these environments back to home, cups to the given positions."""
        env_ids = torch.tensor(ids, device=dev)
        q = robot.data.default_joint_pos[env_ids].clone()
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        robot.set_joint_position_target(q, env_ids=env_ids)
        robot.set_joint_velocity_target(torch.zeros_like(q), env_ids=env_ids)
        for color, cup in cups.items():
            cup_state = torch.zeros((len(ids), 13), device=dev)
            cup_state[:, 3] = 1.0  # quaternion w
            for row, i in enumerate(ids):
                cup_state[row, :2] = torch.tensor(positions[row][color], dtype=torch.float32, device=dev) + origins_t[i, :2]
            cup.write_root_state_to_sim(cup_state, env_ids=env_ids)

    if args.sac or args.sac_eval:
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sac_cups import run as run_sac

        def cup_states():
            picked = (cups[args.picked].data.root_pos_w - origins_t).cpu().numpy()
            target = (cups[args.target].data.root_pos_w - origins_t).cpu().numpy()
            up = [matrix_from_quat(cups[c].data.root_quat_w.cpu().numpy())[:, 2, 2] for c in (args.picked, args.target)]
            return picked, target, up[0], up[1]

        run_sac(SimpleNamespace(
            n=n, device=dev, data=data, sizes=sizes, picked=args.picked, target=args.target, home_ee=home_ee,
            tracker=tracker, static_cam=static_cam, wrist_cam=wrist_cam, ee_poses=ee_poses, control=control,
            reset_envs=reset_envs, place_cups=place_cups, plan=plan, cup_states=cup_states,
            gripper_width=lambda: robot.data.joint_pos[:, fingers].sum(1).cpu().numpy()), args)
        return

    if args.policy:
        evaluate(reset_envs, place_cups, judge, control, ee_poses, home_ee, static_cam, wrist_cam, tracker, data, calib, out)
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    state = [None] * n  # per environment: dict(segments, index, k, warmup, writer, positions, id)
    counter, saved, attempts, closing = 0, 0, 0, []
    pool = ThreadPoolExecutor(16)
    start_time = time.time()

    def start(ids):
        nonlocal counter
        ids = list(ids)
        for i in ids:
            state[i] = {'positions': place_cups(), 'warmup': 10}
        reset_envs(ids, [state[i]['positions'] for i in ids])
        for i in ids:
            counter += 1
            episode_id = '%s_s%d_isaac_episode_%04d' % (stamp, args.seed, counter)
            state[i].update(segments=plan(state[i]['positions']), index=0, k=0, id=episode_id,
                            writer=EpisodeWriter(out / episode_id))

    def finish(i, success):
        nonlocal saved, attempts
        why = judge.last
        s = state[i]
        s['writer'].close()
        attempts += 1
        if success and saved < args.episodes:
            saved += 1
            closing.append((s['writer'], s['id'], s['positions'], True))
        else:
            closing.append((s['writer'], s['id'], s['positions'], False))
        print('[%5.0f s] env %d: %s (%d frames; %s); saved %d / %d of %d attempts' % (
            time.time() - start_time, i, 'SUCCESS' if success else 'failed', len(s['writer'].log['ee']), why, saved,
            args.episodes, attempts), flush=True)

    start(range(n))
    while saved < args.episodes:
        actual = ee_poses()
        targets = np.tile(home_ee, (n, 1, 1))
        gripper = np.ones(n, bool)
        recording = np.zeros(n, bool)
        for i, s in enumerate(state):
            if s['warmup'] > 0:  # just reset: hold at home, not recorded
                s['warmup'] -= 1
                tracker.reset([i], actual[i:i + 1])
                continue
            kind, *seg = s['segments'][s['index']]
            if kind == 'move':
                targets[i], gripper[i] = seg[0][s['k']], seg[1]
                s['k'] += 1
                done = s['k'] >= len(seg[0])
            elif kind == 'wait':
                targets[i], gripper[i] = seg[0], seg[1]
                s['k'] += 1
                done = np.linalg.norm(actual[i, :3, 3] - seg[0][:3, 3]) < 0.004 or s['k'] >= 60
            else:
                targets[i], gripper[i] = seg[0], seg[1]
                s['k'] += 1
                done = s['k'] >= seg[2]
            if done:
                s['index'], s['k'] = s['index'] + 1, 0
            recording[i] = True
        control(targets, gripper)
        rec = np.nonzero(recording)[0]
        if len(rec):
            ee = ee_poses()
            q = robot.data.joint_pos[:, arm].cpu().numpy()
            width = robot.data.joint_pos[:, fingers].sum(1).cpu().numpy()
            frames = {}
            for name, cam in (('cam_2', static_cam), ('cam_1', wrist_cam)):
                frames[name + '_color'] = cam.data.output['rgb'][..., :3].contiguous().cpu().numpy()
                depth = cam.data.output['distance_to_image_plane'][..., 0] * 1000
                frames[name + '_depth'] = torch.nan_to_num(depth, posinf=0).clamp(0, 65535).to(torch.int32).cpu().numpy().astype('<u2')
            jobs = []
            for i in rec:
                writer = state[i]['writer']
                log = writer.log
                log['ee'].append(ee[i])
                log['q'].append(q[i])
                log['width'].append(float(width[i]))
                log['target'].append(targets[i])
                log['open'].append(1.0 if gripper[i] else 0.0)
                for key, proc in writer.procs.items():
                    jobs.append(pool.submit(proc.stdin.write, frames[key][i].tobytes()))
            for job in jobs:
                job.result()
        ended = [i for i in rec if state[i]['index'] >= len(state[i]['segments'])]
        for i in ended:
            finish(i, judge(i))
        if ended and saved < args.episodes:
            start(ended)
        for item in list(closing):  # episodes whose videos are written: keep (lowdim + episode.json) or delete
            writer, episode_id, positions, keep = item
            if writer.done():
                closing.remove(item)
                if keep:
                    save_lowdim(writer.folder, episode_id, writer.log, positions)
                else:
                    shutil.rmtree(writer.folder, ignore_errors=True)
    for s in state:  # unfinished episodes
        if s['writer'] not in [c[0] for c in closing]:
            s['writer'].close()
            closing.append((s['writer'], s['id'], s['positions'], False))
    for writer, episode_id, positions, keep in closing:
        while not writer.done():
            time.sleep(0.1)
        if keep:
            save_lowdim(writer.folder, episode_id, writer.log, positions)
        else:
            shutil.rmtree(writer.folder, ignore_errors=True)
    print('ISAAC_DEMOS_DONE saved %d of %d attempts in %.0f s' % (saved, attempts, time.time() - start_time), flush=True)


main()
app.close()
