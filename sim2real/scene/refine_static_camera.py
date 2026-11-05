#!/usr/bin/env python3
"""Refines the static camera's pose from the demos themselves.

In every stacking demo the gripper closes centred on the cup it picks up and opens centred on
the cup it stacks onto, so at those moments the end effector (forward kinematics, exact) marks
where the cups really are. The static camera's estimate of the same cups (depth point cloud,
estimate_scene.py) is off by its pose error (small, distant tag) plus depth bias along the ray.
On our data a single depth factor explains it best (the depth read ~7% long at 1.2-1.5 m).

    python3 sim2real/scene/refine_static_camera.py --scene ~/franka_scene/<dir>/scene.json \
        --replays ~/franka_scene/<dir>/replays --picked blue --target red
Fits a depth-scale correction for the static camera (points slide along the camera rays: keeps the
camera pose that matches the images) and prints the --depth-scale to rerun estimate_scene.py with;
a planar camera move is fit for comparison. Writes the fit into scene.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def events(replay, close_below=0.075, open_above=0.077):
    """(end-effector pose when the gripper first closes, ... when it opens again afterwards)."""
    w, ee = replay['gripper_width'], replay['o_t_ee']
    closed = np.nonzero(w < close_below)[0]
    if not len(closed):
        return None, None
    grasp = int(closed[0])
    reopened = np.nonzero(w[grasp:] > open_above)[0]
    release = grasp + int(reopened[0]) if len(reopened) else None
    return ee[grasp], (ee[release] if release is not None else None)


def kabsch_2d(source, target):
    sc, tc = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - sc).T @ (target - tc))
    rotation = (u @ vt).T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = (u @ vt).T
    return rotation, tc - rotation @ sc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--replays', required=True, help='folder with episode_<n>_replay.npz (export_replay.py)')
    parser.add_argument('--picked', default='blue', help='cup the gripper picks up')
    parser.add_argument('--target', default='red', help='cup it is stacked onto')
    args = parser.parse_args()
    scene_path = Path(args.scene)
    scene = json.loads(scene_path.read_text())
    observed, truth, labels = [], [], []
    for episode in scene['per_episode']:
        number = episode['episode'].rsplit('_', 1)[1]
        path = Path(args.replays) / ('episode_%s_replay.npz' % number)
        if not path.exists():
            continue
        grasp, release = events(np.load(path))
        cups = {o['color']: o for o in episode['objects']}
        for pose, color in ((grasp, args.picked), (release, args.target)):
            if pose is not None and color in cups:
                observed.append(cups[color]['center_xy'])
                truth.append(pose[:2, 3])
                labels.append((number, color))
    observed, truth = np.array(observed), np.array(truth)
    camera_xy = np.array(scene['static_camera']['T_base_cam_leveled'])[:2, 3]
    before = np.linalg.norm(observed - truth, axis=1)
    # Model 1 (used): the static camera's depth reads long/short by a factor; points slide along the rays.
    # Keeps the camera pose, which matches the images (overlays line up), and needs one parameter.
    rays, rays_true = observed - camera_xy, truth - camera_xy
    current = scene.get('source', {}).get('depth_scale', 1.0)
    factor = float(np.sum(rays * rays_true) / np.sum(rays * rays))
    after_scale = np.linalg.norm(camera_xy + factor * rays - truth, axis=1)
    # Model 2 (comparison only): move the camera (planar rigid fit, three parameters)
    rotation, shift = kabsch_2d(observed, truth)
    after_rigid = np.linalg.norm(observed @ rotation.T + shift - truth, axis=1)
    scene['static_camera']['depth_scale_fit'] = {
        'pairs': len(observed), 'factor_vs_current': factor, 'suggested_depth_scale': current * factor,
        'error_before_m': {'median': float(np.median(before)), 'max': float(np.max(before))},
        'error_after_m': {'median': float(np.median(after_scale)), 'max': float(np.max(after_scale))},
        'rigid_camera_move_error_after_m': float(np.median(after_rigid)),
        'method': 'cup positions seen by the static camera vs the end-effector position when the gripper closed '
                  'on (%s) / opened over (%s) them' % (args.picked, args.target)}
    scene_path.write_text(json.dumps(scene, indent=1))
    print('%d pairs (%s at grasp, %s at release), current depth scale %.3f' % (len(observed), args.picked, args.target, current))
    print('cup position error now:                  median %.1f mm, max %.1f mm' % (np.median(before) * 1000, before.max() * 1000))
    print('with depth x %.3f (scale %.3f):         median %.1f mm, max %.1f mm' % (
        factor, current * factor, np.median(after_scale) * 1000, after_scale.max() * 1000))
    print('(moving the camera instead, 3 parameters: median %.1f mm)' % (np.median(after_rigid) * 1000))
    print('-> rerun estimate_scene.py with --depth-scale %.3f' % (current * factor))

if __name__ == '__main__':
    main()
