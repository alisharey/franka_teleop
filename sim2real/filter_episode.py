#!/usr/bin/env python3
"""Blacks out everything except our table, the robot and the objects, the same way for real and sim.

Every pixel's depth becomes a point in the robot base frame (static camera: its leveled pose and
depth correction from scene/estimate_scene.py + scene/refine_static_camera.py; wrist camera:
end-effector pose at that frame x the hand-eye result) and is kept only inside a box above our
table (table, cups, robot); the lab, the floor and the neighbouring table go black. The filter
itself is sim2real/observation.py (also used live by vla/baku/deploy.py).

    # real (raw episode folder + scene/export_replay.py npz for the end-effector poses)
    filter_episode.py --real ~/franka_teleop_data/<episode> --replay <replay.npz> --out <dir>
    # sim (one isaac/cup_demos.py episode folder)
    filter_episode.py --sim <episode folder> --out <dir>
    # sim training data (all episodes of an isaac/cup_demos.py run, same export layout)
    filter_episode.py --sim-export <root> --out <dir>
Writes <out>/cam_2_filtered.mp4 and cam_1_filtered.mp4 (--sim-export: <out>/<episode>/ with the
filtered cam_*_color.mp4, lowdim.npz and episode.json with "observation": "background_filter").
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim2real.observation import DEFAULT_BOX, DEFAULT_CALIBRATION, DEFAULT_SCENE, STATIC, WRIST, BackgroundFilter  # noqa: E402


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def depth_frames_real(path, width, height):
    process = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', str(path), '-pix_fmt', 'gray16le', '-f', 'rawvideo', '-'],
                               stdout=subprocess.PIPE)
    size = width * height * 2
    while True:
        chunk = process.stdout.read(size)
        if len(chunk) < size:
            break
        yield np.frombuffer(chunk, np.uint16).reshape(height, width)


def filter_camera(bg, camera, color_path, depth_iter, ee_poses, out_path):
    capture = cv2.VideoCapture(str(color_path))
    writer = None
    for index, depth in enumerate(depth_iter):
        ok, bgr = capture.read()
        if not ok:
            break
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (bgr.shape[1], bgr.shape[0]))
        ee = ee_poses[min(index, len(ee_poses) - 1)] if ee_poses is not None else None
        writer.write(bg.apply(camera, bgr, depth, ee))
    if writer:
        writer.release()
    capture.release()


def sim_ee_poses(episode):
    low = np.load(episode / 'lowdim.npz')
    return [np.vstack([np.column_stack([quat_xyzw_to_matrix(q), p]), [0, 0, 0, 1]])
            for p, q in zip(low['ee_pos'], low['ee_quat'])]


def filter_sim_episode(bg, episode, out_folder, names):
    ee = sim_ee_poses(episode)
    for cam in (STATIC, WRIST):
        depth = depth_frames_real(episode / ('%s_depth.mkv' % cam), 640, 480)  # FFV1 16-bit mm, like the recorder
        filter_camera(bg, cam, episode / ('%s_color.mp4' % cam), depth, ee if cam == WRIST else None,
                      out_folder / names[cam])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--real', help='raw recorded episode folder')
    parser.add_argument('--replay', help='export_replay.py npz of that episode (end-effector poses, cam_2 frames)')
    parser.add_argument('--sim', help='isaac/cup_demos.py episode folder')
    parser.add_argument('--sim-export', help='root of isaac/cup_demos.py episodes: writes the same '
                                             'export episodes to --out with filtered camera videos (training data)')
    parser.add_argument('--scene', default=str(DEFAULT_SCENE))
    parser.add_argument('--calibration', default=str(DEFAULT_CALIBRATION))
    parser.add_argument('--box', default=','.join(str(v) for v in DEFAULT_BOX), help='x0,x1,y0,y1,z0,z1 in the base frame [m]')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    box = tuple(float(v) for v in args.box.split(','))
    bg = BackgroundFilter(args.scene, args.calibration, box, sim=not args.real)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filtered = {c: '%s_filtered.mp4' % c for c in (STATIC, WRIST)}
    if args.real:
        episode = Path(args.real)
        meta = json.loads((episode / 'meta.json').read_text())
        replay = np.load(args.replay)
        ee_by_stamp = replay['o_t_ee']
        for cam in (STATIC, WRIST):
            streams = meta['cameras'][cam]['streams']
            ee = None
            if cam == WRIST:
                with open(episode / streams['color']['frames_csv']) as f:
                    stamps = np.array([int(r['stamp_ns']) for r in csv.DictReader(f)], dtype=np.int64)
                nearest = np.clip(np.searchsorted(replay['frame_stamp_ns'], stamps), 0, len(ee_by_stamp) - 1)
                ee = [ee_by_stamp[i] for i in nearest]
            depth_iter = depth_frames_real(episode / streams['depth']['file'], streams['depth']['width'],
                                           streams['depth']['height'])
            filter_camera(bg, cam, episode / streams['color']['file'], depth_iter, ee, out / filtered[cam])
        print('wrote', out / filtered[STATIC], out / filtered[WRIST])
    elif args.sim:
        filter_sim_episode(bg, Path(args.sim), out, filtered)
        print('wrote', out / filtered[STATIC], out / filtered[WRIST])
    else:
        episodes = sorted(p.parent for p in Path(args.sim_export).glob('*/episode.json'))
        for index, episode in enumerate(episodes):
            target = out / episode.name
            if (target / 'episode.json').exists():
                continue
            if not all((episode / ('%s_depth.mkv' % c)).exists() for c in (STATIC, WRIST)):
                print('skip %s (no depth: generator stopped while saving it)' % episode.name)
                continue
            target.mkdir(parents=True, exist_ok=True)
            filter_sim_episode(bg, episode, target, {c: '%s_color.mp4' % c for c in (STATIC, WRIST)})
            for name in ('lowdim.npz',):
                shutil.copy2(episode / name, target / name)
            info = json.loads((episode / 'episode.json').read_text())
            info['observation'] = 'background_filter'
            (target / 'episode.json').write_text(json.dumps(info, indent=1))  # last: marks the episode complete
            print('%d/%d %s' % (index + 1, len(episodes), episode.name), flush=True)


if __name__ == '__main__':
    main()
