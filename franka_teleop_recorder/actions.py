"""Turns a recorded episode into VLA-style actions.

Delta end-effector pose actions at a fixed rate: for each step k (time t0 + k/rate),
  dp_k = p_{k+1} - p_k                          translation, base frame O [m]
  dr_k = log(R_{k+1} R_k^T)                     rotation vector, base frame O [rad]
  gripper_k = 1 (closed / grasp) or 0 (open)    the gripper command state at t_k
so that executing them in order (R <- exp(dr) R, p <- p + dp) retraces the episode.
"""

import json
from pathlib import Path

import numpy as np

STATE_TOPIC = '/teleop/state'


def load_states(episode_dir):
    """The 1 kHz /teleop/state stream of an episode as numpy arrays."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from franka_teleop.msg import TeleopState

    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / 'meta.json').read_text())
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(episode_dir / 'bag'),
                                          storage_id=meta.get('bag', {}).get('storage', 'mcap')),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[STATE_TOPIC]))
    stamps, poses, joints, commands, widths = [], [], [], [], []
    while reader.has_next():
        _, data, _ = reader.read_next()
        state = deserialize_message(data, TeleopState)
        stamps.append(state.header.stamp.sec + state.header.stamp.nanosec * 1e-9)
        poses.append(np.array(state.o_t_ee).reshape(4, 4, order='F'))  # libfranka: column-major
        joints.append(np.array(state.q))
        commands.append(state.gripper_command)
        widths.append(state.gripper_width)
    if not stamps:
        raise ValueError('%s has no %s messages (was gamepad_teleop running?)' % (episode_dir, STATE_TOPIC))
    return {
        'meta': meta,
        't': np.array(stamps),
        'pose': np.array(poses),
        'q': np.array(joints),
        'gripper_command': np.array(commands),
        'gripper_width': np.array(widths),
    }


def telemoma_actions(episode_dir, rate_hz):
    """TeleMoMa's action as recorded on /teleop/quest (input = quest), sampled at
    the same times as delta_pose_actions: t0 + k / rate_hz, t0 = first /teleop/state.

    Returns {'t', 'action' (N x 7: dx dy dz droll dpitch dyaw gripper, gripper
    1 = open), 'movement_enabled', 'target' (N x 4 x 4, row-major)} or None if
    the episode was not recorded with the Quest.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from franka_teleop.msg import QuestState

    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / 'meta.json').read_text())
    if '/teleop/quest' not in meta.get('bag', {}).get('topics', {}):
        return None
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(episode_dir / 'bag'),
                                          storage_id=meta.get('bag', {}).get('storage', 'mcap')),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/teleop/quest', STATE_TOPIC]))
    t0 = None
    stamps, actions, enabled, targets = [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == STATE_TOPIC:
            if t0 is None:
                from franka_teleop.msg import TeleopState
                state = deserialize_message(data, TeleopState)
                t0 = state.header.stamp.sec + state.header.stamp.nanosec * 1e-9
            continue
        quest = deserialize_message(data, QuestState)
        stamps.append(quest.header.stamp.sec + quest.header.stamp.nanosec * 1e-9)
        actions.append(np.array(quest.action))
        enabled.append(quest.movement_enabled)
        targets.append(np.array(quest.target).reshape(4, 4))
    if not stamps:
        return None
    t = np.array(stamps) - (t0 if t0 is not None else stamps[0])
    sample_times = np.arange(0.0, t[-1] + 1e-9, 1.0 / rate_hz)
    index = np.clip(np.searchsorted(t, sample_times), 0, len(t) - 1)
    return {'t': sample_times, 'action': np.array(actions)[index], 'movement_enabled': np.array(enabled)[index],
            'target': np.array(targets)[index]}


def rotation_log(matrix):
    """Rotation vector of a rotation matrix (robust near 0 and pi)."""
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    vee = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    if angle < 1e-7:
        return 0.5 * vee
    if np.pi - angle < 1e-4:
        # R ~ 2 a a^T - I: magnitudes from the diagonal, signs relative to the largest.
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0))
        largest = int(np.argmax(axis))
        for index in range(3):
            if index != largest:
                axis[index] = np.copysign(axis[index], matrix[largest, index] + matrix[index, largest])
        return angle * axis / np.linalg.norm(axis)
    return angle / (2.0 * np.sin(angle)) * vee


def rotation_exp(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    k = np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]) / angle
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def quaternion(matrix):
    """(x, y, z, w) of a rotation matrix."""
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def gripper_closed(command, width, open_width):
    """1 = closed/grasp, 0 = open. Before any command in the session, judge by the width."""
    if command == 2:
        return 1
    if command == 1:
        return 0
    return 1 if width < 0.5 * open_width else 0


def delta_pose_actions(states, rate_hz, open_width=0.08):
    """Samples the episode at rate_hz and returns per-step delta actions."""
    t = states['t'] - states['t'][0]
    sample_times = np.arange(0.0, t[-1] + 1e-9, 1.0 / rate_hz)
    index = np.clip(np.searchsorted(t, sample_times), 0, len(t) - 1)
    earlier = np.clip(index - 1, 0, len(t) - 1)
    index = np.where(np.abs(t[earlier] - sample_times) < np.abs(t[index] - sample_times), earlier, index)
    poses = states['pose'][index]
    positions = poses[:, :3, 3]
    rotations = poses[:, :3, :3]
    steps = len(index) - 1
    dp = positions[1:] - positions[:-1]
    dr = np.array([rotation_log(rotations[k + 1] @ rotations[k].T) for k in range(steps)])
    gripper = np.array([gripper_closed(states['gripper_command'][i], states['gripper_width'][i], open_width)
                        for i in index])
    return {
        'rate_hz': rate_hz,
        'sample_index': index,
        'dp': dp,
        'dr': dr,
        'gripper': gripper,          # steps + 1 values (state at every sample)
        'positions': positions,      # steps + 1 recorded poses, for comparison
        'quaternions': np.array([quaternion(r) for r in rotations]),
        'q_start': states['q'][index[0]],
    }


def integrate(actions):
    """Re-applies the actions from the first recorded pose; reproduces the samples."""
    position = actions['positions'][0].copy()
    rotation = matrix_from_quaternion(actions['quaternions'][0])
    positions = [position.copy()]
    for dp, dr in zip(actions['dp'], actions['dr']):
        position = position + dp
        rotation = rotation_exp(dr) @ rotation
        positions.append(position.copy())
    return np.array(positions), rotation


def matrix_from_quaternion(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_teleop_config(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


REPLAY_CONFIG_KEYS = [
    'robot_ip', 'joystick_path', 'input_timeout_ms', 'max_linear_speed_m_s', 'max_angular_speed_rad_s',
    'max_linear_acceleration_m_s2', 'max_linear_jerk_m_s3', 'max_angular_acceleration_rad_s2',
    'max_angular_jerk_rad_s3', 'workspace_x_m', 'workspace_y_m', 'workspace_z_m', 'allow_detour',
    'gripper_enabled', 'gripper_open_width_m', 'gripper_open_speed_m_s', 'gripper_grasp_width_m',
    'gripper_grasp_speed_m_s', 'gripper_grasp_force_N', 'gripper_grasp_epsilon_inner_m',
    'gripper_grasp_epsilon_outer_m',
]


def write_actions_file(path, episode_id, actions, teleop_config):
    """Plain-text file read by the episode_replay program: key = value header, then CSV rows."""
    lines = ['# franka_teleop replay actions v1 (delta end-effector pose, base frame)',
             'episode = %s' % episode_id,
             'rate_hz = %g' % actions['rate_hz'],
             'steps = %d' % len(actions['dp']),
             'q_start = %s' % ', '.join('%.6f' % v for v in actions['q_start'])]
    for key in REPLAY_CONFIG_KEYS:
        if key not in teleop_config:
            raise KeyError('teleop config is missing %s' % key)
        lines.append('%s = %s' % (key, teleop_config[key]))
    lines.append('[actions]')
    lines.append('step,dx,dy,dz,drx,dry,drz,gripper,rec_x,rec_y,rec_z,rec_qx,rec_qy,rec_qz,rec_qw')
    steps = len(actions['dp'])
    for k in range(steps + 1):
        dp = actions['dp'][k] if k < steps else np.zeros(3)
        dr = actions['dr'][k] if k < steps else np.zeros(3)
        values = [*dp, *dr]
        lines.append('%d,%s,%d,%s' % (
            k, ','.join('%.9f' % v for v in values), actions['gripper'][k],
            ','.join('%.6f' % v for v in [*actions['positions'][k], *actions['quaternions'][k]])))
    Path(path).write_text('\n'.join(lines) + '\n')
