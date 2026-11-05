"""The sim2real observation: camera images with everything but our table, the robot and the objects
blacked out, computed the same way for real and sim frames (training data and live deployment).

Every pixel's depth becomes a point in the robot base frame (static camera: its leveled pose and depth
correction from scene/estimate_scene.py; wrist camera: the end-effector pose x the hand-eye result) and
is kept only inside a box above our table; the lab, the floor and the neighbouring table go black.
Pixels without depth are dropped for the static camera and kept for the wrist camera (its depth has
large holes on near surfaces: the fingers, a held cup, the near table).

    from sim2real.observation import BackgroundFilter
    bg = BackgroundFilter(scene_json, calibration_json, sim=False)
    color = bg.apply('cam_1', color, depth_mm, ee_pose)      # ee_pose: 4x4 O_T_EE (wrist camera only)
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / 'sim2real' / 'scenes' / 'cups_20261024' / 'scene.json'
DEFAULT_CALIBRATION = ROOT / 'config' / 'camera_calibration.json'
DEFAULT_BOX = (-0.35, 1.0, -0.44, 0.45, -0.04, 1.3)  # x0, x1, y0, y1, z0, z1 in the base frame [m]
STATIC, WRIST = 'cam_2', 'cam_1'


class BackgroundFilter:
    def __init__(self, scene=DEFAULT_SCENE, calibration=DEFAULT_CALIBRATION, box=DEFAULT_BOX, sim=False):
        scene = json.loads(Path(scene).read_text())
        calib = json.loads(Path(calibration).read_text())
        self.t_base_static = np.array(scene['static_camera']['T_base_cam_leveled'])
        self.t_ee_wrist = np.array(calib['T_ee_cam_cam_1'])
        self.k = {c: np.array(calib['intrinsics'][c]['K']) for c in (WRIST, STATIC)}
        # the real static camera's depth reads long (refine_static_camera.py); sim depth is exact
        self.depth_scale = {STATIC: 1.0 if sim else scene['source'].get('depth_scale', 1.0), WRIST: 1.0}
        self.box = box
        self._rays = {}

    def rays(self, camera, h, w):
        if (camera, h, w) not in self._rays:
            k = self.k[camera]
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            self._rays[camera, h, w] = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1],
                                                 np.ones_like(u, float)], axis=-1).astype(np.float32)
        return self._rays[camera, h, w]

    def camera_pose(self, camera, ee_pose=None):
        return self.t_base_static if camera == STATIC else np.asarray(ee_pose) @ self.t_ee_wrist

    def mask(self, camera, depth_mm, ee_pose=None):
        """Boolean image: True where the pixel shows our table, the robot or an object."""
        pose = self.camera_pose(camera, ee_pose)
        z = depth_mm.astype(np.float32) * (0.001 * self.depth_scale[camera])
        points = self.rays(camera, *z.shape) * z[..., None] @ pose[:3, :3].T.astype(np.float32) + pose[:3, 3]
        b = self.box
        valid = z > 0.05
        inside = valid & (points[..., 0] > b[0]) & (points[..., 0] < b[1]) & (points[..., 1] > b[2]) & \
            (points[..., 1] < b[3]) & (points[..., 2] > b[4]) & (points[..., 2] < b[5])
        if camera == WRIST:
            # the real wrist depth has large holes with speckle on near surfaces (cups, fingers) that the sim
            # does not have: no flying-pixel step here (it would carve outlines into the real images only);
            # only valid depth outside the box (the lab beyond the table) goes black
            flying = np.zeros_like(valid)
            keep = inside | ~valid
        else:
            # flying pixels: on object outlines the depth mixes foreground and background and lands in between
            smooth = cv2.medianBlur(np.where(valid, z, 0).astype(np.float32), 5)
            flying = valid & (np.abs(z - smooth) > 0.04)
            keep = inside & ~flying
        # fill small holes (depth noise on the table, missing depth) but not across the background
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
        keep &= ~flying
        return cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)

    def apply(self, camera, color, depth_mm, ee_pose=None):
        return np.where(self.mask(camera, depth_mm, ee_pose)[..., None], color, 0).astype(np.uint8)
