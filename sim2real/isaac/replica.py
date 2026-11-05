"""Isaac Lab replica of a recorded cup demo (kinematic replay, rendered side by side with the real cameras).

Scene from sim2real/scene/estimate_scene.py (table, cups) and tools/calibration (camera poses):
  - our table (measured edges, top at z = 0 in the robot base frame) on a floor, the two cups as
    truncated cones with the measured diameters and height at their positions in the episode
  - Isaac Lab's Franka Panda (Isaac Lab has no FR3 model) replaying the recorded joint angles and
    gripper width frame by frame; the moved cup follows the gripper while it is closed around it
  - a static camera at the calibrated (leveled) pose with the real intrinsics, and the wrist camera
    mounted on the hand with the hand-eye result
Writes <out>/replica.mp4 (rows: static, wrist; columns: real, sim) and fk_check.json (Panda forward
kinematics vs the FR3's recorded end-effector pose).

Run through run.sh with SCRIPT=replica.py (Isaac Lab's python, headless, cameras enabled).
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', required=True, help='folder with scene.json, calibration.json, *_replay.npz, real mp4s')
parser.add_argument('--episode', default='0081')
parser.add_argument('--moved-cup', default='blue', help='the cup the gripper carries')
parser.add_argument('--stride', type=int, default=2, help='use every n-th camera frame')
parser.add_argument('--max-frames', type=int, default=0, help='0 = whole episode')
parser.add_argument('--out', required=True)
parser.add_argument('--table-texture', default='', help='table_texture.png from table_texture.py')
parser.add_argument('--robot-usd', default='', help='local copy of panda_instanceable.usd (default: asset server)')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, UsdGeom, Vt  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

HAND_TO_EE = np.eye(4)
HAND_TO_EE[2, 3] = 0.1034  # libfranka's EE frame with the Franka Hand = panda_hand + 10.34 cm along z


def _quat(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return tuple(q / np.linalg.norm(q))


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def add_frustum(stage, path, bottom_d, top_d, height, color, segments=48):
    """Truncated cone, base on z = 0 (cups stand upside down: the rim, the wider end, on the table)."""
    rb, rt = bottom_d / 2, top_d / 2
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    points = [(rb * np.cos(a), rb * np.sin(a), 0.0) for a in angles] + \
             [(rt * np.cos(a), rt * np.sin(a), height) for a in angles] + [(0, 0, 0), (0, 0, height)]
    counts, indices = [], []
    for i in range(segments):
        j = (i + 1) % segments
        counts.append(4)
        indices += [i, j, segments + j, segments + i]
        counts += [3, 3]
        indices += [2 * segments, j, i, 2 * segments + 1, segments + i, segments + j]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr('none')
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45)
    material.func(path + '_material', material)
    sim_utils.bind_visual_material(path, path + '_material')
    return UsdGeom.Xformable(mesh).AddTransformOp()


def add_textured_top(stage, path, x0, x1, y0, y1, texture_file, z=0.0005):
    """Quad on the table top with the real table's texture (table_texture.py: u along x, v along y)."""
    from pxr import Sdf, UsdShade
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z), Gf.Vec3f(x1, y1, z), Gf.Vec3f(x0, y1, z)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    mesh.CreateSubdivisionSchemeAttr('none')
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    material = UsdShade.Material.Define(stage, path + '_material')
    surface = UsdShade.Shader.Define(stage, path + '_material/surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7)
    reader = UsdShade.Shader.Define(stage, path + '_material/uv')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    texture = UsdShade.Shader.Define(stage, path + '_material/texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture_file))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def set_pose(op, pose):
    op.Set(Gf.Matrix4d(pose.T.tolist()))  # USD matrices are row-vector (transposed)


def main():
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = json.loads((data / 'scene.json').read_text())
    calib = json.loads((data / 'calibration.json').read_text())
    replay = np.load(data / ('episode_%s_replay.npz' % args.episode))
    episode = next(e for e in scene['per_episode'] if e['episode'].endswith('episode_' + args.episode))
    t_base_static = np.array(scene['static_camera']['T_base_cam_leveled'])
    t_ee_wrist = np.array(calib['T_ee_cam_cam_1'])
    k_static = np.array(calib['intrinsics']['cam_2']['K'])
    k_wrist = np.array(calib['intrinsics']['cam_1']['K'])

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device))
    sim.set_camera_view([1.8, 1.2, 0.9], [0.3, 0.0, 0.1])
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    table = scene['table']
    table_height = 0.75  # floor below the table top (not measured; only the floor's look depends on it)
    ground = sim_utils.CuboidCfg(size=(8.0, 8.0, 0.02),  # local floor (no download)
                                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.37)))
    ground.func('/World/ground', ground, translation=(0, 0, -table_height - 0.01))
    dome = sim_utils.DomeLightCfg(intensity=700.0, color=(1.0, 1.0, 1.0))
    dome.func('/World/dome', dome)
    sun = sim_utils.DistantLightCfg(intensity=600.0, angle=3.0)
    sun.func('/World/sun', sun, orientation=(0.9, 0.2, 0.3, 0.0))
    lx, ly = table['x_m'][1] - table['x_m'][0], table['y_m'][1] - table['y_m'][0]
    cx, cy = (table['x_m'][1] + table['x_m'][0]) / 2, (table['y_m'][1] + table['y_m'][0]) / 2
    top = sim_utils.CuboidCfg(size=(lx, ly, 0.03), visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.86, 0.86, 0.84), roughness=0.6))
    top.func('/World/table/top', top, translation=(cx, cy, -0.015))
    if args.table_texture:
        add_textured_top(stage, '/World/table/textured_top', table['x_m'][0], table['x_m'][1], table['y_m'][0],
                         table['y_m'][1], Path(args.table_texture).resolve())
    for n, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        leg = sim_utils.CuboidCfg(size=(0.04, 0.04, table_height - 0.03),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)))
        leg.func('/World/table/leg%d' % n, leg, translation=(cx + sx * (lx / 2 - 0.05), cy + sy * (ly / 2 - 0.05),
                                                              -0.03 - (table_height - 0.03) / 2))

    sizes = scene['objects']
    colors = {'blue': (0.05, 0.22, 0.75), 'red': (0.8, 0.05, 0.05)}
    cups, cup_pose = {}, {}
    for obj in episode['objects']:
        if obj['color'] not in colors:
            continue
        s = sizes[obj['color']]
        op = add_frustum(stage, '/World/cup_%s' % obj['color'], s['bottom_diameter_m']['median'],
                         s['top_diameter_m']['median'], s['height_m']['median'], colors[obj['color']])
        pose = np.eye(4)
        pose[:2, 3] = obj['center_xy']
        set_pose(op, pose)
        cups[obj['color']], cup_pose[obj['color']] = op, pose

    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path='/World/robot')
    if args.robot_usd:  # local mirror (mirror_usd.py): the asset server is flaky from compute nodes
        robot_cfg.spawn.usd_path = args.robot_usd
    robot = Articulation(robot_cfg)
    static_cfg = CameraCfg(
        prim_path='/World/static_cam', update_period=0, height=480, width=640, data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=k_static.flatten().tolist(),
                                                               width=640, height=480, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(t_base_static[:3, 3]), rot=_quat(t_base_static[:3, :3]),
                                   convention='ros'))
    t_hand_wrist = HAND_TO_EE @ t_ee_wrist
    wrist_cfg = CameraCfg(
        prim_path='/World/robot/panda_hand/wrist_cam', update_period=0, height=480, width=640, data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=k_wrist.flatten().tolist(),
                                                               width=640, height=480, clipping_range=(0.01, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(t_hand_wrist[:3, 3]), rot=_quat(t_hand_wrist[:3, :3]),
                                   convention='ros'))
    static_cam, wrist_cam = Camera(static_cfg), Camera(wrist_cfg)
    sim.reset()
    hand = robot.find_bodies('panda_hand')[0][0]
    dt = sim.get_physics_dt()

    q, width, ee_rec = replay['q'], replay['gripper_width'], replay['o_t_ee']
    frames = list(range(0, len(q), args.stride))
    if args.max_frames:
        frames = frames[:args.max_frames]
    real = {name: cv2.VideoCapture(str(data / ('episode_%s_%s.mp4' % (args.episode, name)))) for name in ('cam_2', 'cam_1')}
    writer = cv2.VideoWriter(str(out / 'replica_raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30 / args.stride,
                             (1280, 960))
    attached, attach_offset = False, None
    fk_errors = []

    def apply(k):
        position = torch.zeros((1, robot.num_joints), device=robot.device)
        position[0, :7] = torch.tensor(q[k], device=robot.device)
        position[0, 7:9] = float(width[k]) / 2
        robot.write_joint_state_to_sim(position, torch.zeros_like(position))
        robot.set_joint_position_target(position)
        robot.write_data_to_sim()

    apply(0)
    for _ in range(20):  # let the renderer settle
        sim.step()
        robot.update(dt)
        static_cam.update(dt)
        wrist_cam.update(dt)
    last_real = {}
    real_index = 0
    for k in frames:
        apply(k)
        sim.step()
        robot.update(dt)
        hand_pose = np.eye(4)
        hand_pose[:3, 3] = robot.data.body_pos_w[0, hand].cpu().numpy()
        hand_pose[:3, :3] = quat_to_matrix(robot.data.body_quat_w[0, hand].cpu().numpy())
        ee_sim = hand_pose @ HAND_TO_EE
        fk_errors.append(float(np.linalg.norm(ee_sim[:3, 3] - ee_rec[k][:3, 3]) * 1000))
        # the carried cup follows the gripper while it is closed around it (attach once, release on opening)
        cup = cup_pose.get(args.moved_cup)
        if cup is not None:
            grip_center = ee_rec[k][:3, 3]
            if not attached and width[k] < 0.075 and np.linalg.norm(grip_center[:2] - cup[:2, 3]) < 0.06:
                attached, attach_offset = True, np.linalg.inv(ee_rec[k]) @ cup
            elif attached and width[k] > 0.077:
                attached = False
            if attached:
                cup_pose[args.moved_cup] = ee_rec[k] @ attach_offset
                set_pose(cups[args.moved_cup], cup_pose[args.moved_cup])
        sim.render()
        static_cam.update(dt)
        wrist_cam.update(dt)
        sims = {'cam_2': static_cam.data.output['rgb'][0, ..., :3].cpu().numpy(),
                'cam_1': wrist_cam.data.output['rgb'][0, ..., :3].cpu().numpy()}
        while real_index <= k:  # read the real videos up to this frame
            for name, capture in real.items():
                ok, bgr = capture.read()
                if ok:
                    last_real[name] = bgr
            real_index += 1
        rows = []
        for name, label in (('cam_2', 'static'), ('cam_1', 'wrist')):
            real_bgr = last_real.get(name, np.zeros((480, 640, 3), np.uint8))
            sim_bgr = cv2.cvtColor(sims[name].astype(np.uint8), cv2.COLOR_RGB2BGR)
            for image, text in ((real_bgr, 'real %s' % label), (sim_bgr, 'Isaac Lab %s' % label)):
                cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append(np.hstack([real_bgr, sim_bgr]))
        frame = np.vstack(rows)
        cv2.putText(frame, 't = %.1f s   gripper %.0f mm%s' % (replay['t'][k], width[k] * 1000,
                                                              '   carrying %s cup' % args.moved_cup if attached else ''),
                    (12, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        if k == frames[len(frames) // 2]:
            cv2.imwrite(str(out / 'replica_middle_frame.png'), frame)
    writer.release()
    check = {'episode': args.episode, 'frames': len(frames), 'robot_model': 'Isaac Lab Franka Panda (no FR3 asset)',
             'ee_position_error_mm': {'median': float(np.median(fk_errors)), 'max': float(np.max(fk_errors))},
             'note': 'Panda forward kinematics with the recorded FR3 joint angles vs the FR3 O_T_EE recorded by libfranka'}
    (out / 'fk_check.json').write_text(json.dumps(check, indent=1))
    print('REPLICA_DONE', json.dumps(check))


main()
app.close()
