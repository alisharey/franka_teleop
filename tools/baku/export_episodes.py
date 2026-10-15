#!/usr/bin/env python3
"""Recorded episodes -> per-episode training folders for BAKU (or any policy).

For every color frame of the observation camera (cam_1 by default, ~30 Hz) it
writes the robot state and the action labels at that instant, so a policy sees
(images at frame k) -> (action at frame k). The bag (1 kHz /teleop/state and
/teleop/quest) is read here because it needs ROS; images stay in the recorded
MP4s and are decoded later by the dataset builder (on the cluster), which also
picks the resize/crop.

Output, one folder per episode:
    <out>/<episode_id>/lowdim.npz      arrays below, one row per observation frame
    <out>/<episode_id>/cam_*_color.mp4 copied unchanged
    <out>/<episode_id>/episode.json    task, success, frame indices, provenance

lowdim.npz:
    t                    [s] since the first observation frame
    frame_<cam>          frame index into <cam>_color.mp4 nearest in time
    ee_pos, ee_quat      end effector in the base frame (quat x y z w)
    ee_rotvec            same orientation as a rotation vector
    q                    joint angles [rad]
    gripper_width        measured opening [m]
    gripper_open         1 = open, 0 = closed: the last gripper command sent
                         (before any, judged from the width)
    telemoma_action      TeleMoMa action as computed (QuestState.action):
                         [dx dy dz droll dpitch dyaw gripper], base frame,
                         position = target - current [m], rotation = scipy
                         "xyz" Euler of R_target R_current^T [rad], clipped to
                         +-1, gripper 1 = open
    movement_enabled     grip held and the reading fresh: the arm was driven
    action               the label to train on: telemoma_action with the pose
                         part zeroed where the arm was NOT driven (grip
                         released or stale reading), and the gripper = the
                         command actually sent (gripper_open). This is exactly
                         what the teleop executed.
    delta_pose           next-frame delta pose for comparison: [dp (3), dr
                         rotation vector (3)] in the base frame, last row 0

    tools/baku/export.sh --out ~/baku_export ~/franka_teleop_data/2026*_episode_00{14,15,16}
"""

import site
import sys

_user_site = site.getusersitepackages()  # ~/.local numpy 2.x breaks ROS Humble (see scripts/recorder)
sys.path[:] = [entry for entry in sys.path if not entry.startswith(_user_site)]

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from franka_teleop_recorder.actions import quaternion, rotation_log  # noqa: E402

GRIPPER_OPEN, GRIPPER_GRASP = 1, 2
FORMAT = 'franka_teleop_baku_export/1'


def read_frames_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r['frame']) for r in rows]), np.array([int(r['stamp_ns']) for r in rows], dtype=np.int64)


def read_bag(episode_dir, meta):
    """The 1 kHz /teleop/state and /teleop/quest streams (only the fields used here)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from franka_teleop.msg import QuestState, TeleopState

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(episode_dir / 'bag'),
                                          storage_id=meta.get('bag', {}).get('storage', 'mcap')),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/teleop/state', '/teleop/quest']))
    state = {'stamp': [], 'tick': [], 'o_t_ee': [], 'q': [], 'width': [], 'command': []}
    quest = {'stamp': [], 'tick': [], 'action': [], 'movement_enabled': [], 'reading_age': [], 'pose_valid': [],
             'policy_active': []}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/teleop/state':
            m = deserialize_message(data, TeleopState)
            state['stamp'].append(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec)
            state['tick'].append(m.tick)
            state['o_t_ee'].append(np.array(m.o_t_ee).reshape(4, 4, order='F'))  # libfranka: column-major
            state['q'].append(np.array(m.q))
            state['width'].append(m.gripper_width)
            state['command'].append(m.gripper_command)
        else:
            m = deserialize_message(data, QuestState)
            quest['stamp'].append(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec)
            quest['tick'].append(m.tick)
            quest['action'].append(np.array(m.action))
            quest['movement_enabled'].append(m.movement_enabled)
            quest['reading_age'].append(m.reading_age)
            quest['pose_valid'].append(m.pose_valid)
            quest['policy_active'].append(m.policy_active)
    state = {k: np.array(v) for k, v in state.items()}
    quest = {k: np.array(v) for k, v in quest.items()}
    if len(state['stamp']) == 0:
        raise ValueError('no /teleop/state messages')
    if len(quest['stamp']) == 0:
        raise ValueError('no /teleop/quest messages (not recorded with the Quest)')
    return state, quest


def nearest(stamps, query):
    """Index of the stamp nearest to each query (stamps sorted)."""
    index = np.clip(np.searchsorted(stamps, query), 0, len(stamps) - 1)
    earlier = np.clip(index - 1, 0, len(stamps) - 1)
    return np.where(np.abs(stamps[earlier] - query) < np.abs(stamps[index] - query), earlier, index)


def latest_at(stamps, query):
    """Index of the last stamp at or before each query (the value in force then)."""
    return np.clip(np.searchsorted(stamps, query, side='right') - 1, 0, len(stamps) - 1)


def gripper_open_series(command, width, open_width):
    """1 = open, 0 = closed: the last command; before the first one, judged from the width."""
    result = np.where(width > 0.5 * open_width, 1, 0)
    result = np.where(command == GRIPPER_OPEN, 1, result)
    return np.where(command == GRIPPER_GRASP, 0, result)


def export(episode_dir, out_root, observation_camera, timeout_s, open_width, task_override):
    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / 'meta.json').read_text())
    cameras = sorted(meta['cameras'])
    if observation_camera not in cameras:
        raise ValueError('camera %s not in %s' % (observation_camera, cameras))
    state, quest = read_bag(episode_dir, meta)

    frames = {}
    for cam in cameras:
        stream = meta['cameras'][cam]['streams']['color']
        frames[cam] = read_frames_csv(episode_dir / stream['frames_csv'])
    obs_index, obs_stamp = frames[observation_camera]
    # Only frames inside the robot stream (the cameras start a little earlier/later).
    inside = (obs_stamp >= state['stamp'][0]) & (obs_stamp <= state['stamp'][-1])
    obs_index, obs_stamp = obs_index[inside], obs_stamp[inside]
    if len(obs_stamp) < 2:
        raise ValueError('fewer than 2 frames overlap the robot stream')

    s = latest_at(state['stamp'], obs_stamp)
    qi = latest_at(quest['stamp'], obs_stamp)
    pose = state['o_t_ee'][s]
    ee_pos = pose[:, :3, 3]
    ee_quat = np.array([quaternion(r) for r in pose[:, :3, :3]])
    ee_rotvec = np.array([rotation_log(r) for r in pose[:, :3, :3]])
    gripper_open = gripper_open_series(state['command'][s], state['width'][s], open_width)

    telemoma = quest['action'][qi].copy()
    driven = (quest['movement_enabled'][qi] & quest['pose_valid'][qi] & quest['policy_active'][qi]
              & (quest['reading_age'][qi] <= timeout_s))
    action = telemoma.copy()
    action[~driven, :6] = 0.0
    action[:, 6] = gripper_open

    delta = np.zeros((len(s), 6))
    delta[:-1, :3] = ee_pos[1:] - ee_pos[:-1]
    rotations = pose[:, :3, :3]
    delta[:-1, 3:] = np.array([rotation_log(rotations[k + 1] @ rotations[k].T) for k in range(len(s) - 1)])

    arrays = {
        't': (obs_stamp - obs_stamp[0]) * 1e-9,
        'stamp_ns': obs_stamp,
        'ee_pos': ee_pos, 'ee_quat': ee_quat, 'ee_rotvec': ee_rotvec,
        'q': state['q'][s],
        'gripper_width': state['width'][s],
        'gripper_open': gripper_open.astype(np.float64),
        'telemoma_action': telemoma,
        'movement_enabled': driven,
        'action': action,
        'delta_pose': delta,
    }
    for cam in cameras:
        index, stamp = frames[cam]
        arrays['frame_' + cam] = index[nearest(stamp, obs_stamp)]
        arrays['frame_skew_ms_' + cam] = (stamp[nearest(stamp, obs_stamp)] - obs_stamp) * 1e-6

    out = Path(out_root) / meta['id']
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'lowdim.npz', **arrays)
    for cam in cameras:
        shutil.copy2(episode_dir / meta['cameras'][cam]['streams']['color']['file'], out / ('%s_color.mp4' % cam))
    info = {
        'format': FORMAT,
        'episode_id': meta['id'],
        'name': meta.get('name', ''),
        'task': task_override if task_override is not None else meta.get('task', ''),
        'success': meta.get('success'),
        'notes': meta.get('notes', ''),
        'cameras': {cam: {'file': '%s_color.mp4' % cam,
                          'width': meta['cameras'][cam]['streams']['color']['width'],
                          'height': meta['cameras'][cam]['streams']['color']['height'],
                          'serial': meta['cameras'][cam].get('serial')} for cam in cameras},
        'observation_camera': observation_camera,
        'frames': int(len(obs_stamp)),
        'duration_s': float((obs_stamp[-1] - obs_stamp[0]) * 1e-9),
        'fps': float((len(obs_stamp) - 1) / ((obs_stamp[-1] - obs_stamp[0]) * 1e-9)),
        'driven_fraction': float(driven.mean()),
        'action_convention': 'telemoma: [dx dy dz] = target - current [m]; [droll dpitch dyaw] = scipy xyz '
                             'Euler of R_target R_current^T [rad]; base frame; pose part 0 when not driven; '
                             'gripper 1 = open, 0 = closed',
        'teleop_config_text': meta.get('teleop_session', {}).get('config_text', ''),
        'git_commit': meta.get('teleop_session', {}).get('git_commit', ''),
    }
    (out / 'episode.json').write_text(json.dumps(info, indent=1))
    skew = max(np.abs(arrays['frame_skew_ms_' + c]).max() for c in cameras)
    print('%s: %d frames at %.1f Hz, driven %.0f%%, |pose action| max %.3f m / %.3f rad, camera skew max %.1f ms, '
          'task %r' % (meta['id'], len(obs_stamp), info['fps'], 100 * driven.mean(),
                       np.abs(action[:, :3]).max(), np.abs(action[:, 3:6]).max(), skew, info['task']))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episodes', nargs='+', help='episode folders')
    parser.add_argument('--out', required=True, help='output root')
    parser.add_argument('--observation-camera', default='cam_1', help='camera whose frames set the clock')
    parser.add_argument('--task', default=None, help='task text for all episodes (default: each meta.json task)')
    parser.add_argument('--timeout-ms', type=float, default=500.0, help='quest_timeout_ms used while recording')
    parser.add_argument('--gripper-open-width', type=float, default=0.08)
    args = parser.parse_args()
    failed = 0
    for episode in args.episodes:
        try:
            export(episode, args.out, args.observation_camera, args.timeout_ms * 1e-3, args.gripper_open_width,
                   args.task)
        except Exception as error:  # keep going; report at the end
            failed += 1
            print('%s: SKIPPED (%s)' % (episode, error))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
