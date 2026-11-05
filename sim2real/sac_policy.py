"""The SAC actor trained in Isaac Lab (isaac/sac_cups.py) as a deployable policy for vla/baku/deploy.py.

Same interface as BAKU's FrankaPolicy (preprocess, act, reset, rate_hz, pixel_keys, cameras, last), plus
set_robot(O_T_EE, gripper_width) before each act: besides the two background-filtered camera images
(84x84) the actor gets the end-effector position, the gripper width and its own gripper command. It
outputs a target offset (5 cm x [dx dy dz]) and open/close; act() returns our 7-D TeleMoMa action with the
rotation part holding the orientation the episode started with (the sim policy never rotates the
gripper) and the gripper as 1 = open / 0 = closed (hysteresis as in training). Needs only torch + OpenCV.

A run folder: <run>/policy.json {"type": "sac", "task": "..."}, <run>/observation.json (background
filter), <run>/snapshot/<name>.pt (the actor's state_dict).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

IMG = 84
N_PROPRIO = 5   # end-effector position (base frame), gripper width, gripper command
STEP = 0.05     # action 1 = 5 cm target offset
RATE_HZ = 10.0  # 3 control steps of the 30 Hz sim per action


class Encoder(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(6, 32, 3, stride=2), nn.ReLU(), nn.Conv2d(32, 32, 3), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3), nn.ReLU(), nn.Conv2d(32, 32, 3), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            size = self.conv(torch.zeros(1, 6, IMG, IMG)).shape[1]
        self.head = nn.Sequential(nn.Linear(size, features), nn.LayerNorm(features), nn.Tanh())

    def forward(self, images):
        return self.head(self.conv(images))


class Actor(nn.Module):
    """Attribute names match sac_cups.Policy (a skrl Model), so its state_dict loads as is."""

    def __init__(self, num_actions=4):
        super().__init__()
        self.encoder = Encoder()
        self.net = nn.Sequential(nn.Linear(64 + N_PROPRIO, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, num_actions), nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(num_actions))

    def forward(self, images, proprio):
        """images (n, 6, IMG, IMG) in [0, 1], proprio (n, N_PROPRIO) -> mean action (n, 4)."""
        return self.net(torch.cat([self.encoder(images), proprio], 1))


def euler_xyz(r):
    """scipy 'xyz' (extrinsic) of a rotation matrix, as TeleMoMa's action."""
    return np.array([math.atan2(r[2, 1], r[2, 2]), math.asin(max(-1.0, min(1.0, -r[2, 0]))),
                     math.atan2(r[1, 0], r[0, 0])])


def is_sac_run(checkpoint):
    path = Path(checkpoint).parents[1] / 'policy.json'
    return path.exists() and json.loads(path.read_text()).get('type') == 'sac'


class SACPolicy:
    needs_robot = True

    def __init__(self, snapshot, device='cuda', max_steps=600):
        run = Path(snapshot).resolve().parents[1]
        spec = json.loads((run / 'policy.json').read_text())
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.actor = Actor().to(self.device).eval()
        self.actor.load_state_dict(torch.load(snapshot, map_location=self.device))
        self.pixel_keys = ['pixels', 'pixels_egocentric']
        self.cameras = {'pixels': 'cam_2', 'pixels_egocentric': 'cam_1'}
        self.rate_hz = RATE_HZ
        self.max_steps = int(max_steps)
        self.info = {'task_texts': [spec.get('task', '')], 'num_queries': 1}
        self.task_index, self.task_text = 0, spec.get('task', '')
        self.snapshot = str(snapshot)
        self.robot = None
        self.reset()

    def set_task(self, task):
        pass  # one task per SAC run

    def reset(self):
        self.step_index = 0
        self.open = True
        self.hold_rotation = None
        self.last = {}

    def set_robot(self, o_t_ee, gripper_width):
        """The robot state for the next act(): O_T_EE (4x4, base frame) and the measured gripper width [m]."""
        self.robot = (np.asarray(o_t_ee, dtype=np.float64), float(gripper_width))

    def preprocess(self, frames):
        """{key: H x W x 3 RGB (background-filtered)} -> {key: 3 x IMG x IMG uint8}, as in training."""
        return {key: cv2.resize(frames[key], (IMG, IMG), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                for key in self.pixel_keys}

    @torch.no_grad()
    def act(self, frames, preprocessed=False):
        if self.robot is None:
            raise RuntimeError('SACPolicy needs set_robot() before act()')
        obs = frames if preprocessed else self.preprocess(frames)
        pose, width = self.robot
        images = np.concatenate([obs['pixels'], obs['pixels_egocentric']])[None].astype(np.float32) / 255.0
        proprio = np.array([[*pose[:3, 3], width, 1.0 if self.open else 0.0]], np.float32)
        mean = self.actor(torch.from_numpy(images).to(self.device), torch.from_numpy(proprio).to(self.device))
        mean = mean[0].cpu().numpy().astype(np.float64)
        if self.hold_rotation is None:
            self.hold_rotation = pose[:3, :3].copy()
        self.open = True if mean[3] > 0.3 else False if mean[3] < -0.3 else self.open
        action = np.zeros(7)
        action[:3] = np.clip(mean[:3], -1, 1) * STEP
        action[3:6] = euler_xyz(self.hold_rotation @ pose[:3, :3].T)
        action[6] = 1.0 if self.open else 0.0
        self.step_index += 1
        self.last = {'raw_output': mean}
        return action

    def token_columns(self, count=7):
        return [math.nan] * count, [math.nan] * count
