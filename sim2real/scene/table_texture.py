#!/usr/bin/env python3
"""Top-down texture of our table from the static camera, for the sim table.

Every texel of the table top (z = 0, the table rectangle from estimate_scene.py) is projected into
the static camera with the leveled pose and intrinsics, and its color is sampled from the first
frames of the episodes. A texel counts only where the (depth-scaled) depth image sees the table
there (not a cup or the robot in front of it), so the per-texel median over episodes removes the
objects, which stand somewhere else in every episode. What is never visible (behind the robot's
foot, outside the view) is filled by inpainting.

Texture layout: columns run along x (table x_m[0] -> x_m[1]), rows along y from y_m[1] (top row)
to y_m[0]; i.e. UV u = (x - x0) / (x1 - x0), v = (y - y0) / (y1 - y0).

    tools/calibration/.venv/bin/python sim2real/scene/table_texture.py --scene <dir>/scene.json \
        --out <dir>/table_texture.png ~/franka_teleop_data/2026..._episode_*
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def first_frames(episode, cam, count):
    meta = json.loads((episode / 'meta.json').read_text())
    streams = meta['cameras'][cam]['streams']
    w, h = streams['color']['width'], streams['color']['height']
    capture = cv2.VideoCapture(str(episode / streams['color']['file']))
    ok, bgr = capture.read()
    capture.release()
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(episode / streams['depth']['file']), '-frames:v',
                          str(count), '-pix_fmt', 'gray16le', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    depth = np.median(np.frombuffer(raw, np.uint16).reshape(-1, h, w).astype(np.float32), axis=0) * 0.001
    k = np.array(meta['cameras'][cam]['color_camera_info']['k']).reshape(3, 3)
    return bgr, depth, k


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episodes', nargs='+')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--camera', default='cam_2')
    parser.add_argument('--mm-per-texel', type=float, default=2.0)
    parser.add_argument('--tolerance', type=float, default=0.025, help='depth agreement for a texel to count [m]')
    parser.add_argument('--base-radius', type=float, default=0.16, help='repaint the table this close to the robot base')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    scene = json.loads(Path(args.scene).read_text())
    depth_scale = scene['source'].get('depth_scale', 1.0)
    x0, x1 = scene['table']['x_m']
    y0, y1 = scene['table']['y_m']
    step = args.mm_per_texel / 1000
    width, height = int(round((x1 - x0) / step)), int(round((y1 - y0) / step))
    xs = x0 + (np.arange(width) + 0.5) * step
    ys = y1 - (np.arange(height) + 0.5) * step  # row 0 = y1
    gx, gy = np.meshgrid(xs, ys)
    table_points = np.stack([gx, gy, np.zeros_like(gx), np.ones_like(gx)], axis=-1).reshape(-1, 4)

    by_episode = {e['episode']: e for e in scene['per_episode']}
    samples, valid = [], []
    for path in sorted(args.episodes):
        path = Path(path)
        meta_id = json.loads((path / 'meta.json').read_text())['id']
        entry = by_episode.get(meta_id)
        t_base_cam = np.array(entry['T_base_cam_leveled'] if entry else scene['static_camera']['T_base_cam_leveled'])
        bgr, depth, k = first_frames(path, args.camera, 5)
        depth *= depth_scale
        cam_points = (np.linalg.inv(t_base_cam) @ table_points.T).T[:, :3]
        z = cam_points[:, 2]
        u = k[0, 0] * cam_points[:, 0] / z + k[0, 2]
        v = k[1, 1] * cam_points[:, 1] / z + k[1, 2]
        inside = (z > 0.1) & (u >= 0) & (u < bgr.shape[1] - 1) & (v >= 0) & (v < bgr.shape[0] - 1)
        map_u = np.where(inside, u, -1).astype(np.float32).reshape(height, width)
        map_v = np.where(inside, v, -1).astype(np.float32).reshape(height, width)
        color = cv2.remap(bgr, map_u, map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        seen_depth = cv2.remap(depth, map_u, map_v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        ok = inside.reshape(height, width) & (seen_depth > 0) & \
            (np.abs(seen_depth - z.reshape(height, width)) < args.tolerance)
        samples.append(color)
        valid.append(ok)
    samples, valid = np.array(samples, np.float32), np.array(valid)
    stack = np.where(valid[..., None], samples, np.nan)
    with np.errstate(all='ignore'):
        import warnings
        warnings.simplefilter('ignore', RuntimeWarning)
        texture = np.nanmedian(stack, axis=0)
    coverage = valid.sum(axis=0)
    hole = (coverage < 2) | (np.hypot(gx, gy) < args.base_radius)  # the robot's foot (edge, shadow, LED ring)
    texture = np.nan_to_num(texture, nan=0).clip(0, 255).astype(np.uint8)
    texture = cv2.inpaint(texture, hole.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    out = Path(args.out)
    cv2.imwrite(str(out), texture)
    cv2.imwrite(str(out.with_name(out.stem + '_coverage.png')), (np.clip(coverage / max(1, coverage.max()), 0, 1) * 255).astype(np.uint8))
    info = {'texture': out.name, 'x_m': [x0, x1], 'y_m': [y0, y1], 'mm_per_texel': args.mm_per_texel,
            'size_px': [width, height], 'layout': 'columns along +x, rows from y_max (top) to y_min',
            'episodes': len(args.episodes), 'filled_fraction': float(hole.mean())}
    out.with_suffix('.json').write_text(json.dumps(info, indent=1))
    print('texture %dx%d (%.0f mm/texel) from %d episodes; %.0f%% never visible (inpainted) -> %s' % (
        width, height, args.mm_per_texel, len(args.episodes), 100 * hole.mean(), out))


if __name__ == '__main__':
    main()
