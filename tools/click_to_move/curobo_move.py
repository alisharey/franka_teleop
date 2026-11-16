#!/usr/bin/env python3
"""Click a point in the static camera's view and the gripper goes there, cuRobo (v0.9) planning the whole arm
around what you identified, and nothing else:
  the table: "Identify table", click its 4 corners in the camera view (each the median depth around the click),
    "Save table": a 5 cm thick box under the plane through them, as large as the rectangle they span, goes into
    cuRobo (saved in config/table_plane.json, used from then on);
  the obstacle: "Select object", click it; EdgeTAM (tracker.py) tracks it, and its mask's depth (minus the robot's
    pixels and anything within TABLE_BAND_M of the table) is fused into cuRobo's TSDF mapper (1 cm voxels, rebuilt
    from every new frame) and turned into an ESDF that the planner checks every robot sphere against.
--inflate grows the robot's spheres by that much, the same as inflating every obstacle.
The cuRobo world is drawn over the camera image on the right (the table box, the obstacle's voxels coloured by
height, the robot's spheres). It is not updated while the arm moves (the arm hides what is behind it).

Robot model: cuRobo's Franka (arm, hand, fingers) plus spheres around the wrist camera and its mount.

Motion: cuRobo plans the joints; the TCP path of that plan is followed by gamepad_teleop --input policy (its own
IK, speed limits, workspace fence and contact limit), the way click_to_move.py moves. The hand follows the
planned path; the elbow can differ from the plan.

Runs in tools/click_to_move/curobo/.venv (cuRobo needs torch >= 2.5). Started by curobo_move.sh.

    http://localhost:8781
"""
import argparse
import json
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

# torch.compile in-process: its worker processes outlived the page when it was stopped
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
import torch  # noqa: E402

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.perception import Mapper, MapperCfg
from curobo.scene import Cuboid, Scene
from curobo.types import CameraObservation, GoalToolPose, JointState, Pose
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

from click_to_move import ROOT, Mover, Ros, Table, workspace_fence
from tracker import Tracker

TABLE_FILE = ROOT / 'config' / 'table_plane.json'
TABLE_BAND_M = 0.015     # depth this close to the table plane is the table (the box), not an obstacle
ROBOT_MARGIN_M = 0.03    # depth this close to the robot's spheres is the robot
VOXEL_M, ESDF_VOXEL_M = 0.01, 0.02
# The mapped and planned region in the base frame (centre, size) [m]
GRID_CENTER, GRID_EXTENT = (0.45, 0.0, 0.35), (1.3, 1.4, 0.9)
F_T_HAND = np.array([[math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0, 0],  # panda_hand_joint: -45 deg about z
                     [math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
DEFAULT_F_T_EE = F_T_HAND @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])


class Plane(Table):
    """A table plane from a point on it and its normal; rect: the table's extent on it (from its corners), or None."""

    def __init__(self, origin, up, rect=None):
        self.origin = np.asarray(origin, float)
        self.up = np.asarray(up, float) / np.linalg.norm(up)
        e1 = np.array([1.0, 0, 0]) - self.up[0] * self.up
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.up, self.e1)
        self.rect = rect  # cv2.minAreaRect in this plane's xy: ((cx, cy), (w, h), angle in degrees)

    def corners(self):
        return self.point(cv2.boxPoints(self.rect).astype(float), np.zeros(4))

    def box_frame(self):
        """Centre (on the plane) and axes (along w, along h, up) of the rectangle."""
        (cx, cy), _, angle = self.rect
        a = math.radians(angle)
        ex = math.cos(a) * self.e1 + math.sin(a) * self.e2
        return self.point([cx, cy], 0.0)[0], ex, np.cross(self.up, ex), self.up


def table_from_points(points):
    """Least-squares plane through the clicked points, the smallest rectangle around them on it."""
    pts = np.asarray(points, float)
    centre = pts.mean(0)
    normal = np.linalg.svd(pts - centre, full_matrices=False)[2][2]
    plane = Plane(centre, normal if normal[2] > 0 else -normal)
    plane.rect = cv2.minAreaRect(plane.xy(pts).astype(np.float32))
    return plane


def resample(points, step=0.01):
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return np.array(out)


def pose(T):
    return Pose.from_matrix(torch.as_tensor(np.asarray(T), dtype=torch.float32, device='cuda').unsqueeze(0))


class World:
    """cuRobo: the Franka model, the TSDF/ESDF mapper fed from one fixed depth camera, the table box, the planner."""

    def __init__(self, table, T_base_cam, T_ee_cam, image_shape, inflate):
        self.T_base_cam = T_base_cam
        self.lock = threading.Lock()  # one cuRobo call at a time
        robot = load_yaml(join_path(get_robot_configs_path(), 'franka.yml'))
        kin = robot['robot_cfg']['kinematics']
        kin['collision_sphere_buffer'] = inflate
        kin['collision_spheres']['panda_hand'] = kin['collision_spheres']['panda_hand'] + self._camera_spheres(T_ee_cam)
        self.robot_cfg = robot
        h, w = image_shape
        self.mapper = Mapper(MapperCfg(
            extent_meters_xyz=GRID_EXTENT, extent_esdf_meters_xyz=GRID_EXTENT,
            grid_center=torch.tensor(GRID_CENTER, dtype=torch.float32), voxel_size=VOXEL_M,
            esdf_voxel_size=ESDF_VOXEL_M, truncation_distance=VOXEL_M * 4, depth_minimum_distance=0.15,
            depth_maximum_distance=2.0, minimum_tsdf_weight=0.1, num_cameras=1,
            image_height=h, image_width=w, block_size=2))
        self.rgb = torch.zeros((1, h, w, 3), dtype=torch.uint8, device='cuda')
        self.grid = self.mapper.compute_esdf()
        self.set_table(table, build=False)
        started = time.monotonic()
        self.planner = MotionPlanner(MotionPlannerCfg.create(
            robot=robot, scene_model=self._scene(), collision_cache={'cuboid': 2}, graph_planner_algorithm='rrt',
            use_cuda_graph=True))
        self.planner.warmup(enable_graph=True, num_warmup_iterations=3)
        print('cuRobo ready in %.0f s' % (time.monotonic() - started), flush=True)
        self.rays = None

    @staticmethod
    def _camera_spheres(T_ee_cam):
        """Spheres (panda_hand frame) around the wrist D435 (90 x 25 x 25 mm, along the camera's x) and its mount."""
        T = np.linalg.inv(F_T_HAND) @ DEFAULT_F_T_EE @ T_ee_cam  # hand <- camera
        body = [T[:3, :3] @ np.array([x, 0.0, -0.012]) + T[:3, 3] for x in (-0.055, -0.02, 0.015, 0.05)]
        mount = (T[:3, 3] + np.array([0.0, 0.0, 0.02])) / 2  # halfway to the hand's side
        return [{'center': [float(v) for v in c], 'radius': 0.022} for c in body + [mount]]

    def _scene(self):
        return Scene(cuboid=[] if self.table_box is None else [self.table_box], voxel=[self.grid])

    def set_table(self, table, build=True):
        """table: its plane (depth within TABLE_BAND_M of it is never an obstacle); a box only when it has a rect."""
        self.table = table
        self.table_box = None
        if getattr(table, 'rect', None) is not None:
            centre, ex, ey, up = table.box_frame()
            T = np.eye(4)
            T[:3, :3] = np.stack([ex, ey, up], 1)
            T[:3, 3] = centre - 0.025 * up
            q = Pose.from_matrix(torch.tensor(T, dtype=torch.float32).unsqueeze(0)).quaternion[0].tolist()
            (w, h) = table.rect[1]
            self.table_box = Cuboid(name='table', dims=[float(w), float(h), 0.05], pose=[*T[:3, 3].tolist(), *q])
        if build:
            with self.lock:
                self.planner.update_world(self._scene())

    # ---------------------------------------------------------------- perception
    def joint_state(self, q):
        q = torch.as_tensor(np.asarray(q)[:7], dtype=torch.float32, device='cuda').unsqueeze(0)
        return JointState.from_position(q, joint_names=self.planner.joint_names)

    def robot_spheres(self, q):
        with self.lock:
            s = self.planner.kinematics.compute_kinematics(self.joint_state(q)).robot_spheres
        s = s.reshape(-1, 4)
        return s[s[:, 3] > 0]

    def integrate(self, depth_mm, K, q, mask=None):
        """One depth frame into the map: robot, table (and, with a mask, everything else) removed."""
        started = time.monotonic()
        d = torch.as_tensor(depth_mm.astype(np.float32), device='cuda') * 0.001
        if self.rays is None or self.rays.shape[:2] != d.shape:
            v, u = torch.meshgrid(torch.arange(d.shape[0], device='cuda'), torch.arange(d.shape[1], device='cuda'),
                                  indexing='ij')
            self.rays = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u, dtype=torch.float64)], -1).float()
        R = torch.as_tensor(self.T_base_cam[:3, :3], dtype=torch.float32, device='cuda')
        t = torch.as_tensor(self.T_base_cam[:3, 3], dtype=torch.float32, device='cuda')
        p = (self.rays * d[..., None]) @ R.T + t
        up = torch.as_tensor(self.table.up, dtype=torch.float32, device='cuda')
        origin = torch.as_tensor(self.table.origin, dtype=torch.float32, device='cuda')
        keep = (d > 0.15) & (((p - origin) @ up) > TABLE_BAND_M)
        if mask is not None:
            keep &= torch.as_tensor(mask, device='cuda')
        spheres = self.robot_spheres(q)
        idx = keep.nonzero(as_tuple=True)
        if len(idx[0]):
            near = (torch.cdist(p[idx], spheres[:, :3]) - spheres[:, 3]).min(dim=1).values < ROBOT_MARGIN_M
            keep[idx[0][near], idx[1][near]] = False
        depth = torch.where(keep, d, torch.zeros_like(d))
        obs = CameraObservation(depth_image=depth.unsqueeze(0), rgb_image=self.rgb, pose=pose(self.T_base_cam),
                                intrinsics=torch.as_tensor(K, dtype=torch.float32, device='cuda').unsqueeze(0))
        with self.lock:
            self.mapper.reset()  # the newest frame is the world: nothing stale stays behind a moved object
            self.mapper.integrate(obs)
            self.grid = self.mapper.compute_esdf()  # the planner reads this grid's tensor in place
            voxels = self.mapper.extract_occupied_voxels(surface_only=True, max_points=40000).centers
        torch.cuda.synchronize()
        return voxels.reshape(-1, 3).cpu().numpy(), (time.monotonic() - started) * 1000

    def reset(self):
        with self.lock:
            self.mapper.reset()
            self.grid = self.mapper.compute_esdf()

    # ---------------------------------------------------------------- planning
    def plan(self, q, T_goal_tcp, f_t_ee):
        """Joint plan to the TCP goal; returns (TCP path (N, 3), joint path (N, 7)) or raises."""
        hand_T_ee = np.linalg.inv(F_T_HAND) @ f_t_ee
        goal = GoalToolPose.from_poses({'panda_hand': pose(T_goal_tcp @ np.linalg.inv(hand_T_ee))})
        with self.lock:
            result = self.planner.plan_pose(goal, self.joint_state(q), max_attempts=5)
            if result is None or not bool(result.success.any()):
                status = getattr(result, 'status', None)
                raise RuntimeError('NO PATH FOUND: cuRobo has no collision-free plan to that target%s' % (
                    '' if status is None else ' (%s)' % status))
            traj = result.get_interpolated_plan()  # all joints, the locked fingers too
            idx = [traj.joint_names.index(n) for n in self.planner.joint_names]
            qs = traj.position.reshape(-1, traj.position.shape[-1])[:, idx].contiguous()
            hand = self.planner.kinematics.compute_kinematics(JointState.from_position(
                qs, joint_names=self.planner.joint_names)).tool_poses.to_dict()['panda_hand']
        H = hand.get_matrix().reshape(-1, 4, 4).cpu().numpy()
        tcp = (H @ hand_T_ee)[:, :3, 3]
        return resample(tcp), qs.cpu().numpy()


class Demo:
    def __init__(self, args):
        calib = json.loads(Path(args.calibration).read_text())
        self.T_base_cam = np.array(calib['T_base_cam_' + args.static])
        self.calibration_session = Path(calib.get('session', '?')).name
        t = json.loads(TABLE_FILE.read_text()) if TABLE_FILE.exists() else {}
        if len(t.get('clicked', [])) >= 3:
            self.table = table_from_points(t['clicked'])
            self.table_source = 'identified %s' % t.get('saved', '')
        else:  # no table in cuRobo; the board's plane only sets "up" and what depth counts as the table
            self.table, self.table_source = Table(calib), None
        self.up = self.table.up
        self.fence = workspace_fence(args.teleop_config)
        self.static = args.static
        self.ros = Ros([args.static])
        self.inflate = args.inflate
        self.mover = Mover(self.ros, args.port_udp, args.speed)
        self.world = self.tracker = None  # built in the background (cuRobo warms up for ~30 s); the page is up meanwhile
        self.loading = 'waiting for the %s camera' % self.static
        threading.Thread(target=self._start, args=(np.array(calib['T_ee_cam_' + args.wrist]),), daemon=True).start()
        self.hover_m = args.hover
        self.point = None
        self.plan = None             # {'path', 'rotation', 'ms'} or {'error', 'ms'}
        self.voxels = np.zeros((0, 3))
        self.table_points = []       # "Identify table" clicks (base frame)
        self.map_ms = None
        self.lock = threading.Lock()

    def _start(self, T_ee_cam):
        try:
            while self.ros.frame(self.static)[1] is None or self.ros.frame(self.static)[2] is None:
                time.sleep(0.1)
            self.loading = 'cuRobo warming up (about 30 s)'
            print(self.loading, flush=True)
            self.world = World(self.table, self.T_base_cam, T_ee_cam, self.ros.frame(self.static)[1].shape, self.inflate)
            self.tracker = Tracker(self.ros.rgb_frame)  # after cuRobo's CUDA graphs are captured
            self.loading = 'EdgeTAM warming up (about 15 s)'
            try:
                self._warm_up()
            except Exception as error:  # only slower first selection
                print('warm-up failed: %r' % error, flush=True)
            self.loading = None
            threading.Thread(target=self._map_loop, daemon=True).start()
        except Exception as error:
            self.loading = 'cuRobo did not start: %r (see the log)' % error
            raise

    def _warm_up(self):
        """One hidden selection (the image centre) through tracking and mapping: the first one after a start takes
        ~15 s (kernels compiled on first use), which made the first real selection look broken."""
        while not self.tracker.ready and not self.tracker.error:
            time.sleep(0.2)
        _, depth, K = self.ros.frame(self.static)
        robot = self.ros.robot()
        if self.tracker.error or robot is None:
            return
        self.tracker.start(self.static, depth.shape[1] / 2, depth.shape[0] / 2)
        deadline = time.monotonic() + 60
        while self.tracker.latest() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        result = self.tracker.latest()
        if result is not None:
            self.world.integrate(result['depth'], result['K'], robot['q'], result['mask'])
        self.tracker.clear()
        self.world.reset()

    def ready(self):
        if self.loading:
            raise RuntimeError(self.loading)

    # ---------------------------------------------------------------- map
    def _map_loop(self):
        last = None
        while True:
            robot = self.ros.robot()
            color, depth, K = self.ros.frame(self.static)
            stamp = self.ros.stamp.get(self.static)
            if self.mover.moving or robot is None or depth is None or K is None or stamp == last:
                time.sleep(0.02)
                continue
            last = stamp
            result = self.tracker.latest()
            if result is None or result['mask'].shape != depth.shape:  # no object: an empty world (but the table)
                if len(self.voxels):
                    self.world.reset()
                    with self.lock:
                        self.voxels = np.zeros((0, 3))
                time.sleep(0.02)
                continue
            mask = cv2.dilate(result['mask'].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            try:
                voxels, ms = self.world.integrate(depth, K, robot['q'], mask)
            except Exception as error:  # keep mapping after a bad frame
                print('mapping failed: %r' % error, flush=True)
                time.sleep(0.5)
                continue
            with self.lock:
                self.voxels = voxels
                self.map_ms = ms if self.map_ms is None else 0.8 * self.map_ms + 0.2 * ms
            time.sleep(0.05)

    def clear_object(self):
        self.ready()
        self.tracker.clear()
        self.world.reset()
        with self.lock:
            self.plan, self.voxels = None, np.zeros((0, 3))

    def table_point(self, fx, fy):
        point, _, _ = self.pixel_point(fx, fy)
        with self.lock:
            self.table_points.append(point)
        return self.table_preview()

    def table_preview(self):
        """The plane through the clicked table points (least squares), or None below 3 points."""
        with self.lock:
            pts = np.array(self.table_points)
        out = {'points': len(pts)}
        if len(pts) >= 3:
            plane = table_from_points(pts)
            out.update(plane=plane, max_mm=round(float(np.abs(plane.height(pts)).max()) * 1000, 1),
                       size_cm=[round(float(v) * 100) for v in plane.rect[1]],
                       tilt_deg=round(math.degrees(math.acos(min(1.0, plane.up[2]))), 2))
        return out

    def table_clear(self):
        with self.lock:
            self.table_points = []

    def table_save(self):
        self.ready()
        preview = self.table_preview()
        if 'plane' not in preview:
            raise RuntimeError("click the table's corners first (at least 3)")
        plane = preview.pop('plane')
        with self.lock:
            pts = [p.tolist() for p in self.table_points]
            self.table_points = []
        TABLE_FILE.write_text(json.dumps({'origin': plane.origin.tolist(), 'up': plane.up.tolist(), 'clicked': pts,
                                          'max_residual_mm': preview['max_mm'],
                                          'saved': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
        self.table, self.up, self.table_source = plane, plane.up, 'identified %s' % time.strftime('%Y-%m-%d %H:%M:%S')
        self.world.set_table(plane)
        self.world.reset()
        return preview

    # ---------------------------------------------------------------- targets
    def pixel_point(self, fx, fy):
        _, depth, K = self.ros.frame(self.static)
        if depth is None or K is None:
            raise RuntimeError('no depth or intrinsics from %s yet' % self.static)
        h, w = depth.shape
        u, v = int(round(fx * (w - 1))), int(round(fy * (h - 1)))
        patch = depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
        patch = patch[patch > 0]
        if patch.size == 0:
            raise RuntimeError('no depth at that pixel: click somewhere else')
        z = float(np.median(patch)) * 0.001
        p = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        return self.T_base_cam[:3, :3] @ p + self.T_base_cam[:3, 3], u, v

    def goal(self, robot):
        """TCP goal: the hover height above the clicked point along the table's normal, pointing into the table,
        the current rotation about that axis kept."""
        z = -self.up
        x = robot['ee'][:3, 0] - np.dot(robot['ee'][:3, 0], z) * z
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(z, [0.0, 1.0, 0.0])
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
        T[:3, 3] = self.point + self.hover_m * self.up
        return T

    def make_plan(self):
        self.ready()
        robot = self.ros.robot()
        if robot is None:
            raise RuntimeError('no robot state: is the teleop running?')
        goal = self.goal(robot)
        g = goal[:3, 3]
        if not all(lo - 1e-3 <= g[i] <= hi + 1e-3 for i, (lo, hi) in enumerate(self.fence.values())):
            raise RuntimeError('the target is outside the workspace fence %s' % self.fence)
        started = time.monotonic()
        try:
            path, qs = self.world.plan(robot['q'], goal, robot.get('f_t_ee', DEFAULT_F_T_EE))
            plan = {'path': path, 'rotation': goal[:3, :3],
                    'ms': (time.monotonic() - started) * 1000}
        except RuntimeError as error:
            plan = {'error': str(error), 'ms': (time.monotonic() - started) * 1000}
        with self.lock:
            self.plan = plan
        return plan

    def click(self, fx, fy):
        point, _, _ = self.pixel_point(fx, fy)
        self.point = point
        if self.mover.moving:
            raise RuntimeError('wait until the move has finished')
        self.make_plan()
        return {'point': point.round(4).tolist(), 'plan': self.plan_summary()}

    def select_object(self, fx, fy):
        self.ready()
        _, u, v = self.pixel_point(fx, fy)
        self.world.reset()
        self.tracker.start(self.static, u, v)
        return {'tracking': self.static}

    def go(self, hover_m):
        if self.point is None:
            raise RuntimeError('click a target point first')
        self.hover_m = max(0.02, float(hover_m))
        robot = self.ros.robot()
        if robot is None:
            raise RuntimeError('no robot state: is the teleop running?')
        plan = self.make_plan()  # again: the hover, the arm or the obstacle may have changed since the click
        if 'error' in plan:
            raise RuntimeError(plan['error'])
        off = float(np.linalg.norm(plan['path'][-1] - robot['ee'][:3, 3]))
        if off < 0.005:
            raise RuntimeError('Already there: the gripper is %.1f mm from that target. Click a new target.' % (off * 1000))
        self.mover.follow(plan['path'], plan['rotation'], lambda position: None)
        return {'plan': self.plan_summary()}

    def plan_summary(self):
        plan = self.plan
        if plan is None:
            return None
        if 'error' in plan:
            return {'error': plan['error'], 'ms': round(plan['ms'])}
        return {'length_m': round(float(np.sum(np.linalg.norm(np.diff(plan['path'], axis=0), axis=1))), 3),
                'ms': round(plan['ms'])}

    def status(self):
        robot = self.ros.robot()
        t = self.tracker
        if t is None:
            t = type('T', (), {'ready': False, 'error': None, 'cam': None, 'fps': None, 'ms': None})()
        with self.lock:
            voxels, map_ms = len(self.voxels), self.map_ms
        return {'point': None if self.point is None else self.point.round(4).tolist(), 'hover_m': self.hover_m,
                'moving': self.mover.moving, 'message': self.mover.message, 'loading': self.loading,
                'table': self.table_source, 'inflate_cm': round(self.inflate * 100, 1),
                'calibration': self.calibration_session, 'plan': self.plan_summary(),
                'map': {'voxels': voxels, 'ms': None if map_ms is None else round(map_ms), 'frozen': self.mover.moving},
                'tracker': {'ready': t.ready, 'error': t.error, 'tracking': t.cam is not None,
                            'fps': None if t.fps is None else round(t.fps, 1), 'ms': None if t.ms is None else round(t.ms)},
                'robot': None if robot is None else {'ee_xyz': robot['ee'][:3, 3].round(4).tolist()}}

    # ---------------------------------------------------------------- drawing
    def _projector(self, K):
        cam_from_base = np.linalg.inv(self.T_base_cam)

        def px(p):
            q = np.atleast_2d(p) @ cam_from_base[:3, :3].T + cam_from_base[:3, 3]
            z = np.maximum(q[:, 2], 1e-6)
            uv = np.stack([K[0, 0] * q[:, 0] / z + K[0, 2], K[1, 1] * q[:, 1] / z + K[1, 2]], 1)
            ok = (q[:, 2] > 0.05) & np.all(np.abs(uv) < 4000, axis=1)
            return uv, z, ok
        return px

    @staticmethod
    def _draw_table(img, px, plane, color):
        """The table's rectangle (the top of its box) with a 10 cm grid."""
        if getattr(plane, 'rect', None) is None:
            return
        centre, ex, ey, _ = plane.box_frame()
        w, h = plane.rect[1]
        lines = [[centre + a * ex - h / 2 * ey, centre + a * ex + h / 2 * ey] for a in np.arange(-w / 2, w / 2 + 1e-6, 0.1)]
        lines += [[centre - w / 2 * ex + b * ey, centre + w / 2 * ex + b * ey] for b in np.arange(-h / 2, h / 2 + 1e-6, 0.1)]
        for a, b in lines:
            uv, _, ok = px(np.linspace(a, b, 20))
            if ok.sum() > 1:
                cv2.polylines(img, [uv[ok].astype(np.int32)], False, color, 1, cv2.LINE_AA)
        uv, _, ok = px(plane.corners())
        if ok.all():
            cv2.polylines(img, [uv.astype(np.int32)], True, color, 2, cv2.LINE_AA)

    def _draw_table_points(self, img, px):
        with self.lock:
            pts = list(self.table_points)
        if not pts:
            return
        preview = self.table_preview()
        if 'plane' in preview:
            self._draw_table(img, px, preview['plane'], (0, 230, 255))
        uv, _, ok = px(np.array(pts))
        for (u, v), good in zip(uv, ok):
            if good:
                cv2.circle(img, (int(u), int(v)), 6, (0, 230, 255), 2, cv2.LINE_AA)

    def _draw_plan(self, img, px):
        with self.lock:
            plan = self.plan
        if plan and 'path' in plan:
            uv, _, ok = px(plan['path'])
            pts = uv[ok].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, (255, 0, 255), 2, cv2.LINE_AA)
        if self.point is not None:
            for p, color, label in ((self.point, (0, 0, 255), 'point'), (self.point + self.hover_m * self.up, (0, 200, 255), 'goal')):
                uv, _, ok = px(p)
                if ok[0]:
                    c = tuple(int(x) for x in uv[0])
                    cv2.drawMarker(img, c, color, cv2.MARKER_CROSS, 22, 2)
                    cv2.putText(img, label, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def view(self, which):
        color, _, K = self.ros.frame(self.static)
        if color is None or K is None:
            img = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(img, 'waiting for %s' % self.static, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return img
        px = self._projector(K)
        if which == 'camera':
            img = color.copy()
            result = None if self.tracker is None else self.tracker.latest()
            if result is not None and result['mask'].shape == img.shape[:2]:
                contours, _ = cv2.findContours(result['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
            self._draw_plan(img, px)
            self._draw_table_points(img, px)
            return img
        img = (color * 0.35).astype(np.uint8)
        self._draw_table(img, px, self.table, (170, 170, 170))  # cuRobo's table box (its top)
        with self.lock:
            voxels = self.voxels
        if len(voxels):
            uv, z, ok = px(voxels)
            h = self.table.height(voxels)
            order = np.argsort(-z)  # far first
            colors = cv2.applyColorMap(np.clip(h / 0.4 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
            for i in order:
                if ok[i]:
                    r = max(1, int(K[0, 0] * VOXEL_M / 2 / z[i]))
                    c = uv[i].astype(int)
                    cv2.rectangle(img, tuple(c - r), tuple(c + r), tuple(int(x) for x in colors[i]), -1)
        robot = self.ros.robot()
        if robot is not None and self.world is not None:
            try:
                spheres = self.world.robot_spheres(robot['q']).cpu().numpy()
                uv, z, ok = px(spheres[:, :3])
                for (u, v), zz, good, r in zip(uv, z, ok, spheres[:, 3]):
                    if good:
                        cv2.circle(img, (int(u), int(v)), max(1, int(K[0, 0] * (r + self.inflate) / zz)), (80, 220, 80), 1, cv2.LINE_AA)
            except Exception:
                pass
        self._draw_plan(img, px)
        self._draw_table_points(img, px)
        return img


PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>cuRobo move</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#f6f6f4; --panel:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#deded8; --accent:#2f6fde; --bad:#c8372d; --warn:#c77700; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --panel:#1f1f1d; --text:#ecece8; --muted:#9a9a93; --line:#34342f; } }
body { margin:0; background:var(--bg); color:var(--text); font:14px system-ui, sans-serif; }
main { max-width:1500px; margin:0 auto; padding:16px; display:grid; gap:14px; }
.views { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:14px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; }
img { width:100%; display:block; border-radius:6px; background:#000; } img.click { cursor:crosshair; }
.bar { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
button, select { padding:8px 14px; border-radius:8px; border:1px solid var(--line); background:var(--panel); color:var(--text); font:inherit; cursor:pointer; }
button.go { background:var(--accent); border-color:var(--accent); color:#fff; } button.stop { background:var(--bad); border-color:var(--bad); color:#fff; font-weight:650; }
button.armed { background:var(--warn); border-color:var(--warn); color:#fff; }
.seg button { border-radius:0; margin-left:-1px; } .seg button:first-child { border-radius:8px 0 0 8px; } .seg button:last-child { border-radius:0 8px 8px 0; }
.seg button.on { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
input { width:60px; padding:6px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--text); font:inherit; }
.muted { color:var(--muted); } h2 { margin:0 0 8px; font-size:14px; }
</style></head><body><main>
<div class="panel bar"><b>A click on the camera view</b>
  <span class="seg"><button data-mode="target">sets the target</button><button data-mode="obstacle">selects the obstacle</button><button data-mode="table">adds a table corner</button></span></div>
<div class="panel bar">
  <button id="clear">Clear obstacle</button>
  <button id="tsave">Save table (from the corners)</button><button id="tclear">Clear corners</button>
  <span>hover <input id="hover" type="number" step="1" min="2" value="10"> cm</span>
  <button class="go" id="go">Go</button><button class="stop" id="stop">Stop</button></div>
<div class="panel bar"><span id="msg">Click a target point in the camera view.</span></div>
<div class="panel bar muted" id="info">&nbsp;</div>
<div class="views">
  <div class="panel"><h2>static camera (click here)</h2><img class="click" id="cam" src="/stream/camera"></div>
  <div class="panel"><h2>cuRobo world</h2><img src="/stream/world"></div>
</div>
<div class="panel muted">Right: everything cuRobo plans around: the table box (grey, its top with a 10 cm grid) and the selected
object's voxels from the newest depth frame (colour = height above the table, blue low, red 40 cm+). Yellow: your "Identify
table" corners and the table they span; the robot's collision spheres grown by the inflation (green). Magenta: the planned path of the gripper's centre. Red / orange
crosses: the clicked point and where the gripper's centre goes. The map does not update while the arm moves.
Keep the robot's stop button in reach.</div>
</main><script>
const $ = (id) => document.getElementById(id);
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const d = await r.json(); if (!r.ok || d.ok === false) throw new Error(d.error || r.statusText); return d;
}
function planText(p) { return !p ? '' : p.error ? `Plan: ${p.error} (${p.ms} ms).` : `Plan: ${p.length_m} m path, planned in ${p.ms} ms.`; }
const HINT = {target: 'Click where the gripper should go (it plans at once; then Go).',
  obstacle: 'Click on the obstacle: EdgeTAM tracks it and it goes into cuRobo.',
  table: "Click the table's 4 corners (on the top, just inside the edges), then Save table."};
let mode = 'target';
function setMode(m) { mode = m; for (const b of document.querySelectorAll('.seg button')) b.classList.toggle('on', b.dataset.mode === m);
  $('msg').textContent = HINT[m]; }
for (const b of document.querySelectorAll('.seg button')) b.addEventListener('click', () => setMode(b.dataset.mode));
setMode('target');
$('clear').addEventListener('click', () => api('/api/object/clear', {}).then(() => { $('msg').textContent = 'Obstacle cleared: cuRobo has only the table.'; })
  .catch((e) => { $('msg').textContent = e.message; }));
function tableText(r) { return r.points < 3 ? `${r.points} corner(s): click the table's 4 corners.`
  : `${r.points} corners: a ${r.size_cm[0]} x ${r.size_cm[1]} cm table (yellow), corners off its plane by up to ${r.max_mm} mm. ${r.points < 4 ? 'Click the last corner, or ' : ''}Save table.`; }
$('tclear').addEventListener('click', () => api('/api/table/clear', {}).then(() => { $('msg').textContent = 'Corners cleared.'; }).catch(() => {}));
$('tsave').addEventListener('click', () => api('/api/table/save', {}).then((r) => { setMode('target');
  $('msg').textContent = `Table saved and in cuRobo: ${r.size_cm[0]} x ${r.size_cm[1]} cm. Now click a target.`; }).catch((e) => { $('msg').textContent = e.message; }));
$('cam').addEventListener('click', async (e) => {
  const b = $('cam').getBoundingClientRect(), at = {x: (e.clientX - b.left) / b.width, y: (e.clientY - b.top) / b.height};
  try {
    if (mode === 'table') { $('msg').textContent = tableText(await api('/api/table/point', at)); return; }
    if (mode === 'obstacle') { await api('/api/object', at); setMode('target');
                               $('msg').textContent = 'Obstacle selected (cyan outline; its voxels on the right in a few seconds). Now click a target.'; return; }
    $('msg').textContent = 'Planning...'; const r = await api('/api/click', at);
    $('msg').textContent = `Target x ${r.point[0].toFixed(3)} y ${r.point[1].toFixed(3)} z ${r.point[2].toFixed(3)} m. ${planText(r.plan)} Check the path, then Go.`;
  } catch (err) { $('msg').textContent = err.message; }
});
$('go').addEventListener('click', async () => { try { $('msg').textContent = 'Planning...';
  const r = await api('/api/go', {hover_m: Number($('hover').value) / 100}); $('msg').textContent = 'Moving: ' + planText(r.plan); } catch (e) { $('msg').textContent = e.message; } });
$('stop').addEventListener('click', () => api('/api/stop', {}).catch(() => {}));
let lastMove = null;
async function poll() {
  try { const s = await api('/api/status'), t = s.tracker, m = s.map;
    let info = s.loading ? `Starting: ${s.loading}... ` : '';
    info += !s.robot ? 'No robot state: is the teleop running? ' : '';
    info += s.table ? `Table: ${s.table}. ` : 'No table in cuRobo: Identify table. ';
    info += !t.ready ? (t.error || 'EdgeTAM loading... ') : t.tracking ? `Obstacle: ${m.voxels} voxels (${m.ms ?? '-'} ms per frame${m.frozen ? ', frozen during the move' : ''}); EdgeTAM ${t.fps ?? '-'} fps (${t.ms ?? '-'} ms per frame). ` : 'No obstacle in cuRobo: Select obstacle. ';
    info += `Inflation ${s.inflate_cm} cm.`;
    $('info').textContent = info;
    // the move's message only while moving or when it is new (it stays 'arrived ...' until the next move)
    if (lastMove !== null && (s.moving || s.message !== lastMove)) $('msg').textContent = s.message;
    lastMove = s.message;
  } catch (e) {}
  setTimeout(poll, 300);
}
poll();
</script></body></html>'''


def serve(demo, port):
    page = PAGE.encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, data, code=200):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path == '/api/status':
                self._json(dict(demo.status(), ok=True))
            elif self.path in ('/stream/camera', '/stream/world'):
                which = self.path[8:]
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        jpg = cv2.imencode('.jpg', demo.view(which), [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpg) + jpg + b'\r\n')
                        time.sleep(0.07 if which == 'camera' else 0.15)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._json({'ok': False, 'error': 'not found'}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            try:
                if self.path == '/api/click':
                    return self._json(dict(demo.click(float(body['x']), float(body['y'])), ok=True))
                if self.path == '/api/object':
                    return self._json(dict(demo.select_object(float(body['x']), float(body['y'])), ok=True))
                if self.path == '/api/object/clear':
                    demo.clear_object()
                    return self._json({'ok': True})
                if self.path == '/api/table/point':
                    r = demo.table_point(float(body['x']), float(body['y']))
                    r.pop('plane', None)
                    return self._json(dict(r, ok=True))
                if self.path == '/api/table/clear':
                    demo.table_clear()
                    return self._json({'ok': True})
                if self.path == '/api/table/save':
                    return self._json(dict(demo.table_save(), ok=True))
                if self.path == '/api/go':
                    return self._json(dict(demo.go(body.get('hover_m', demo.hover_m)), ok=True))
                if self.path == '/api/stop':
                    demo.mover.stop()
                    return self._json({'ok': True})
                self._json({'ok': False, 'error': 'not found'}, 404)
            except (RuntimeError, KeyError, ValueError) as error:
                self._json({'ok': False, 'error': str(error)}, 409)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    print('cuRobo move: http://localhost:%d (calibration %s)' % (port, demo.calibration_session), flush=True)
    server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8781)
    p.add_argument('--port-udp', type=int, default=47820, help='policy_port of config/teleop.conf')
    p.add_argument('--calibration', default=str(ROOT / 'config' / 'camera_calibration.json'))
    p.add_argument('--teleop-config', default=str(ROOT / 'config' / 'teleop.conf'), help='for its workspace fence')
    p.add_argument('--wrist', default='cam_1', help='for the wrist camera spheres (its view is not used)')
    p.add_argument('--static', default='cam_2')
    p.add_argument('--hover', type=float, default=0.10, help='default height above the clicked point [m]')
    p.add_argument('--speed', type=float, default=0.6, help='fraction of the teleop speed caps (0.25 m/s, 0.4 rad/s)')
    p.add_argument('--inflate', type=float, default=0.02, help="the robot's collision spheres grown by [m]")
    args = p.parse_args()
    demo = Demo(args)

    def quit_(*_):  # set after rclpy.init, which otherwise takes SIGINT/SIGTERM and keeps the server running
        demo.mover.stop()
        demo.mover.hold()
        os._exit(0)
    signal.signal(signal.SIGTERM, quit_)
    signal.signal(signal.SIGINT, quit_)
    serve(demo, args.port)


if __name__ == '__main__':
    main()
