#!/usr/bin/env python3
"""Table and objects from recorded episodes + the camera calibration (for a sim replica).

For each episode, the first frames of the static (third-person) camera, before
the arm moves, become a point cloud in the robot base frame (depth + intrinsics
+ T_base_cam from tools/calibration). Then:

  table    RANSAC plane near z = 0 (height and tilt relative to the robot base);
           edges from where the plane is seen; on the side of the neighbouring
           table the edge is the teleop workspace limit + margin (the two tables
           touch, so the plane alone cannot separate them)
  objects  points standing on our table inside the workspace, grouped on a 1 cm
           grid; per object: color (hue), height, bottom/top diameter (circle fit
           to the visible side), position
  camera   the AprilTag's pose in each episode vs the calibration (did the camera move?)

Writes <out>/scene.json (averaged sizes, per-episode positions) and, for the first
episode, segmentation.png (table / objects / background) and table_only.png (the
image with everything but our table and the objects blacked out).

    tools/calibration/.venv/bin/python sim2real/scene/estimate_scene.py \
        --out ~/franka_scene/cups_20261024 ~/franka_teleop_data/20261024_*
(work folder; the finished scene is copied to sim2real/scenes/<name>/, see sim2real/README.md)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'calibration'))
from calibrate_cameras import TagDetector, angle_deg, inverse, mean_pose  # noqa: E402


def read_workspace(path):
    text = Path(path).read_text()
    out = {}
    for axis in 'xyz':
        m = re.search(r'^workspace_%s_m\s*=\s*([-\d.]+)\s*,\s*([-\d.]+)' % axis, text, re.M)
        out[axis] = (float(m.group(1)), float(m.group(2)))
    return out


def read_frames(episode, cam, count):
    """First `count` color (BGR) and depth (uint16 mm) frames."""
    meta = json.loads((episode / 'meta.json').read_text())
    streams = meta['cameras'][cam]['streams']
    w, h = streams['color']['width'], streams['color']['height']
    capture = cv2.VideoCapture(str(episode / streams['color']['file']))
    colors = []
    for _ in range(count):
        ok, bgr = capture.read()
        if ok:
            colors.append(bgr)
    capture.release()
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(episode / streams['depth']['file']),
                          '-frames:v', str(count), '-pix_fmt', 'gray16le', '-f', 'rawvideo', '-'],
                         capture_output=True, check=True).stdout
    depths = np.frombuffer(raw, np.uint16).reshape(-1, h, w)
    return colors, depths, np.array(meta['cameras'][cam]['color_camera_info']['k']).reshape(3, 3), meta


def back_project(depth_m, K):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def ransac_plane(points, iterations=400, threshold=0.008, rng=np.random.default_rng(0)):
    best, best_count = None, -1
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if abs(normal[2]) < 0.9:  # the table is roughly horizontal
            continue
        d = -normal @ sample[0]
        count = int(np.sum(np.abs(points @ normal + d) < threshold))
        if count > best_count:
            best, best_count = (normal, d), count
    normal, d = best
    inliers = points[np.abs(points @ normal + d) < threshold]
    centroid = inliers.mean(axis=0)  # refine with least squares (SVD)
    normal = np.linalg.svd(inliers - centroid)[2][-1]
    if normal[2] < 0:
        normal = -normal
    return normal, -normal @ centroid


def circle_fit(xy):
    """Kasa least-squares circle: center, radius."""
    a = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    c = np.linalg.lstsq(a, b, rcond=None)[0]
    center = c[:2] / 2
    return center, float(np.sqrt(c[2] + center @ center))


def color_name(bgr_pixels):
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = np.median(hsv[:, 0]), np.median(hsv[:, 1]), np.median(hsv[:, 2])
    if v < 70:
        return 'black', (h, s, v)
    if s < 50:
        return 'white', (h, s, v)
    if h < 10 or h > 165:
        return 'red', (h, s, v)
    if 90 <= h <= 135:
        return 'blue', (h, s, v)
    return 'other', (h, s, v)


def analyze(episode, args, calib, workspace, detector, keep_images=False):
    colors, depths, K, meta = read_frames(episode, args.camera, args.frames)
    depth = np.median(depths.astype(np.float32), axis=0) * 0.001 * args.depth_scale  # metres, median over frames
    depth[(depths == 0).sum(axis=0) > len(depths) // 2] = 0
    bgr = colors[0]
    t_base_cam = np.array(calib['T_base_cam_' + args.camera])
    pts_cam = back_project(depth, K)
    valid = depth > 0.15
    pts = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]

    # table plane: points near z = 0 in front of / around the robot
    region = valid & (np.abs(pts[..., 2]) < 0.08) & (pts[..., 0] > -0.3) & (pts[..., 0] < 1.4) & (np.abs(pts[..., 1]) < 1.2)
    normal, d = ransac_plane(pts[region][::4])
    raw_plane = {'normal': normal.tolist(), 'd': float(d), 'height_at_base_m': float(-d / normal[2]),
                 'tilt_deg': float(np.degrees(np.arccos(normal[2])))}
    level = np.eye(4)
    if args.level_table:
        # The robot base stands on the table, so the table is z = 0 and horizontal in the base frame. A small
        # rotation error of the static camera (small, distant tag) tilts and shifts the fitted plane; correct it by
        # rotating about the tag (keeps the wrist calibration's tag position) and shifting in z.
        pivot = np.array(calib['T_base_tag'])[:3, 3].copy()
        pivot -= (pivot @ normal + d) * normal  # tag projected onto the fitted plane
        axis = np.cross(normal, [0, 0, 1])
        angle = np.arcsin(np.clip(np.linalg.norm(axis), -1, 1))
        rotation = cv2.Rodrigues(axis / (np.linalg.norm(axis) + 1e-12) * angle)[0] if angle > 1e-9 else np.eye(3)
        level[:3, :3] = rotation
        level[:3, 3] = pivot - rotation @ pivot
        level[2, 3] -= pivot[2]  # the plane through the pivot goes to z = 0
        t_base_cam = level @ t_base_cam
        pts = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
        region = valid & (np.abs(pts[..., 2]) < 0.05) & (pts[..., 0] > -0.3) & (pts[..., 0] < 1.4) & (np.abs(pts[..., 1]) < 1.2)
        normal, d = ransac_plane(pts[region][::4])
    height = pts @ normal + d  # signed height above the table plane
    on_plane = valid & (np.abs(height) < 0.008)

    # table extent from the plane inliers on a 1 cm grid, then the neighbour-table side cut by the workspace
    grid = 0.01
    xs, ys = pts[on_plane][:, 0], pts[on_plane][:, 1]
    x_lo, x_hi = np.percentile(xs, 0.5), np.percentile(xs, 99.5)
    in_ws_x = (xs > workspace['x'][0]) & (xs < workspace['x'][1])
    y_lo, y_hi = np.percentile(ys[in_ws_x], 0.5), np.percentile(ys[in_ws_x], 99.5)

    # objects: above our table, inside the workspace (+margin), away from the robot base
    m = args.margin
    above = valid & (height > args.object_min_height) & (height < 0.30)
    inside = (pts[..., 0] > workspace['x'][0] - m) & (pts[..., 0] < workspace['x'][1] + m) & \
             (pts[..., 1] > workspace['y'][0] - m) & (pts[..., 1] < workspace['y'][1] + m)
    near_base = np.hypot(pts[..., 0], pts[..., 1]) < args.base_radius
    candidate = above & inside & ~near_base
    gx = np.floor((pts[..., 0] + 1.0) / grid).astype(int)
    gy = np.floor((pts[..., 1] + 1.0) / grid).astype(int)
    occupancy = np.zeros((300, 300), np.uint8)
    cx, cy = gx[candidate], gy[candidate]
    keep = (cx >= 0) & (cx < 300) & (cy >= 0) & (cy < 300)
    occupancy[cx[keep], cy[keep]] = 255
    occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_labels, labels = cv2.connectedComponents(occupancy)
    pixel_label = np.zeros(depth.shape, int)
    ok = candidate & (gx >= 0) & (gx < 300) & (gy >= 0) & (gy < 300)
    pixel_label[ok] = labels[gx[ok], gy[ok]]
    objects = []
    for label in range(1, n_labels):
        mask = pixel_label == label
        if mask.sum() < args.min_pixels:
            continue
        p, hgt = pts[mask], height[mask]
        top = float(np.percentile(hgt, 98))
        if top < args.object_min_size:  # flat things: the tag sheet, the seam to the next table
            continue
        saturated = mask & (cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1] > 70)
        name, hsv = color_name(bgr[saturated] if saturated.sum() > 0.2 * mask.sum() else bgr[mask])

        def width_at(lo, hi):
            # silhouette width (= diameter for a round object) of the rows whose points are in [lo, hi] m height
            band = mask & (height >= lo) & (height <= hi)
            widths = []
            for v in np.unique(np.nonzero(band)[0]):
                us = np.nonzero(band[v])[0]
                if len(us) >= 3:
                    widths.append((us.max() - us.min() + 1) * np.median(depth[v, us]) / K[0, 0])
            return float(np.median(widths)) if len(widths) >= 3 else float('nan')
        bottom_d = width_at(args.object_min_height, args.object_min_height + 0.02)
        top_d = width_at(top - 0.02, top)
        center = p[:, :2].mean(axis=0)
        rgb_mean = bgr[mask].reshape(-1, 3)[:, ::-1].mean(axis=0)  # sRGB 0-255, all object pixels (shading included)
        objects.append({'color': name, 'hsv': [float(v) for v in hsv], 'rgb_mean': [float(v) for v in rgb_mean],
                        'pixels': int(mask.sum()), 'height_m': top,
                        'bottom_diameter_m': bottom_d, 'top_diameter_m': top_d,
                        'center_xy': [float(v) for v in center], 'mask_label': label})

    # camera check: the tag now vs at calibration
    dets = [dd for c in colors[:5] for dd in detector.detect(c, K) if dd['id'] == calib.get('tag_id', 4)]
    camera_shift = None
    if dets:
        tag_now = t_base_cam @ mean_pose([dd['pose'] for dd in dets])
        t_base_tag = np.array(calib['T_base_tag'])
        camera_shift = [float(np.linalg.norm(tag_now[:3, 3] - t_base_tag[:3, 3]) * 1000),
                        angle_deg(t_base_tag[:3, :3].T @ tag_now[:3, :3])]

    result = {'episode': meta['id'], 'task': meta.get('task'), 'raw_plane': raw_plane,
              'T_base_cam_leveled': t_base_cam.tolist(), 'leveling': {
                  'rotation_deg': angle_deg(level[:3, :3]), 'translation_mm': (level[:3, 3] * 1000).tolist()},
              'plane_normal': normal.tolist(), 'plane_d': float(d),
              'table_height_at_base_m': float(-d / normal[2]), 'table_tilt_deg': float(np.degrees(np.arccos(normal[2]))),
              'table_seen_x': [float(x_lo), float(x_hi)], 'table_seen_y_within_workspace_x': [float(y_lo), float(y_hi)],
              'objects': objects, 'tag_vs_calibration_mm_deg': camera_shift}
    if keep_images:
        # our table only: plane points (looser threshold for the mask) inside the table rectangle; the side
        # toward the neighbouring table is cut at the workspace limit + margin
        y_cut_lo = workspace['y'][0] - args.table_margin if y_lo < workspace['y'][0] - 0.2 else y_lo - 0.02
        y_cut_hi = workspace['y'][1] + args.table_margin if y_hi > workspace['y'][1] + 0.2 else y_hi + 0.02
        on_plane = valid & (np.abs(height) < 0.015) & (pts[..., 0] > -args.table_back) & (pts[..., 0] < x_hi + 0.02) & \
            (pts[..., 1] > y_cut_lo) & (pts[..., 1] < y_cut_hi)
        on_plane = cv2.morphologyEx(on_plane.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
        seg = bgr.copy()
        seg[on_plane] = (0.5 * seg[on_plane] + 0.5 * np.array([60, 200, 60])).astype(np.uint8)
        palette = {'blue': (255, 120, 0), 'red': (0, 0, 255), 'black': (40, 40, 40), 'white': (230, 230, 230),
                   'other': (0, 200, 255)}
        object_mask = np.zeros(depth.shape, bool)
        for obj in objects:
            mask = pixel_label == obj['mask_label']
            object_mask |= mask
            seg[mask] = (0.4 * seg[mask] + 0.6 * np.array(palette[obj['color']])).astype(np.uint8)
            ys_, xs_ = np.nonzero(mask)
            cv2.putText(seg, '%s h%.0f d%.0f/%.0f mm' % (obj['color'], obj['height_m'] * 1000,
                                                         obj['bottom_diameter_m'] * 1000, obj['top_diameter_m'] * 1000),
                        (int(xs_.min()), int(ys_.min()) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        result['_images'] = (bgr, seg, on_plane, object_mask, pts, height)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episodes', nargs='+')
    parser.add_argument('--calibration', default=str(ROOT / 'config' / 'camera_calibration.json'))
    parser.add_argument('--teleop-config', default=str(ROOT / 'config' / 'teleop.conf'))
    parser.add_argument('--camera', default='cam_2', help='static camera')
    parser.add_argument('--frames', type=int, default=5, help='first frames averaged per episode')
    parser.add_argument('--margin', type=float, default=0.08, help='objects may lie this far outside the workspace')
    parser.add_argument('--base-radius', type=float, default=0.13, help='ignore points this close to the robot base')
    parser.add_argument('--min-pixels', type=int, default=150)
    parser.add_argument('--object-min-height', type=float, default=0.025,
                        help='points this far above the table count as objects (depth noise ~1 cm at 1.5 m)')
    parser.add_argument('--object-min-size', type=float, default=0.04, help='objects lower than this are ignored')
    parser.add_argument('--depth-scale', type=float, default=1.0,
                        help='static camera depth correction (refine_static_camera.py fits it from the demos)')
    parser.add_argument('--no-level-table', dest='level_table', action='store_false',
                        help='do not level the static camera on the table plane')
    parser.add_argument('--table-margin', type=float, default=0.10, help='table edge beyond the workspace on the neighbour side')
    parser.add_argument('--table-back', type=float, default=0.30, help='table back edge behind the base [m] (not visible)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    calib = json.loads(Path(args.calibration).read_text())
    workspace = read_workspace(args.teleop_config)
    detector = TagDetector('tagStandard41h12', calib.get('tag_size_m', 0.0556), -1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for k, episode in enumerate(sorted(args.episodes)):
        r = analyze(Path(episode), args, calib, workspace, detector, keep_images=(k == 0))
        if k == 0:
            bgr, seg, on_plane, object_mask, pts, height = r.pop('_images')
            cv2.imwrite(str(out / 'segmentation.png'), seg)
            keep = (on_plane | object_mask)
            cv2.imwrite(str(out / 'table_only.png'), np.where(keep[..., None], bgr, 0).astype(np.uint8))
        results.append(r)
        print('%s: raw table h %+.1f mm tilt %.2f deg -> leveled by %.2f deg | %s | tag vs calibration %s' % (
            r['episode'][-12:], r['raw_plane']['height_at_base_m'] * 1000, r['raw_plane']['tilt_deg'],
            r['leveling']['rotation_deg'],
            ', '.join('%s h%.0f d%.0f/%.0f @(%.2f,%.2f)' % (o['color'], o['height_m'] * 1000, o['bottom_diameter_m'] * 1000,
                                                           o['top_diameter_m'] * 1000, *o['center_xy']) for o in r['objects']),
            None if r['tag_vs_calibration_mm_deg'] is None else '%.1f mm / %.2f deg' % tuple(r['tag_vs_calibration_mm_deg'])))

    # aggregate
    def stat(values):
        values = np.array([v for v in values if np.isfinite(v)])
        return {'median': float(np.median(values)), 'p10': float(np.percentile(values, 10)),
                'p90': float(np.percentile(values, 90)), 'n': int(len(values))} if len(values) else None
    objects = {}
    for color in sorted({o['color'] for r in results for o in r['objects']}):
        obs = [o for r in results for o in r['objects'] if o['color'] == color]
        objects[color] = {'episodes_seen': len(obs), 'height_m': stat([o['height_m'] for o in obs]),
                          'bottom_diameter_m': stat([o['bottom_diameter_m'] for o in obs]),
                          'top_diameter_m': stat([o['top_diameter_m'] for o in obs]),
                          'x_range_m': [float(np.min([o['center_xy'][0] for o in obs])), float(np.max([o['center_xy'][0] for o in obs]))],
                          'y_range_m': [float(np.min([o['center_xy'][1] for o in obs])), float(np.max([o['center_xy'][1] for o in obs]))]}
    seen_x = np.median([r['table_seen_x'] for r in results], axis=0)
    seen_y = np.median([r['table_seen_y_within_workspace_x'] for r in results], axis=0)
    # the neighbouring table: on the side where the seen surface runs well past the workspace (> 0.2 m), it is the
    # next table; cut there at the workspace limit + margin. The other side keeps the seen edge.
    y_lo = workspace['y'][0] - args.table_margin if seen_y[0] < workspace['y'][0] - 0.2 else seen_y[0]
    y_hi = workspace['y'][1] + args.table_margin if seen_y[1] > workspace['y'][1] + 0.2 else seen_y[1]
    scene = {
        'source': {'calibration': args.calibration, 'episodes': [r['episode'] for r in results], 'camera': args.camera,
                   'depth_scale': args.depth_scale},
        'robot_base': {'note': 'origin of the base frame (libfranka O); the robot stands at (0, 0)',
                       'table_surface_z_at_base_m': float(np.median([r['table_height_at_base_m'] for r in results]))},
        'table': {'surface_z_m': float(np.median([r['table_height_at_base_m'] for r in results])),
                  'tilt_deg': float(np.median([r['table_tilt_deg'] for r in results])),
                  'x_m': [-args.table_back, float(seen_x[1])],
                  'y_m': [float(y_lo), float(y_hi)],
                  'seen_x_m': seen_x.tolist(), 'seen_y_m': seen_y.tolist(),
                  'note': 'x back edge: hidden behind the robot, assumed --table-back behind the base; front edge seen; y edges: seen surface, cut at the '
                          'workspace limit + %.2f m where it continues onto the neighbouring table' % args.table_margin},
        'workspace': workspace,
        'objects': objects,
        'static_camera': {
            'T_base_cam_calibrated': calib['T_base_cam_' + args.camera],
            'T_base_cam_leveled': mean_pose([np.array(r['T_base_cam_leveled']) for r in results]).tolist(),
            'leveling_rotation_deg': stat([r['leveling']['rotation_deg'] for r in results]),
            'raw_table_tilt_deg': stat([r['raw_plane']['tilt_deg'] for r in results]),
            'raw_table_height_at_base_m': stat([r['raw_plane']['height_at_base_m'] for r in results])},
        'camera_check_tag_vs_calibration': stat([r['tag_vs_calibration_mm_deg'][0] for r in results
                                                 if r['tag_vs_calibration_mm_deg']]),
        'per_episode': results,
    }
    (out / 'scene.json').write_text(json.dumps(scene, indent=1))
    print('\ntable: surface z %+.1f mm, tilt %.2f deg, x %.2f..%.2f m, y %.2f..%.2f m (seen y %.2f..%.2f)' % (
        scene['table']['surface_z_m'] * 1000, scene['table']['tilt_deg'], *scene['table']['x_m'], *scene['table']['y_m'], *seen_y))
    for color, o in objects.items():
        print('%-6s seen in %d episodes: height %.0f mm, bottom d %.0f mm, top d %.0f mm, x %.2f..%.2f, y %.2f..%.2f' % (
            color, o['episodes_seen'], o['height_m']['median'] * 1000,
            (o['bottom_diameter_m'] or {'median': np.nan})['median'] * 1000,
            (o['top_diameter_m'] or {'median': np.nan})['median'] * 1000, *o['x_range_m'], *o['y_range_m']))
    cc = scene['camera_check_tag_vs_calibration']
    print('camera check (tag vs calibration): median %.1f mm' % cc['median'] if cc else 'camera check: tag not seen')
    print('wrote', out / 'scene.json', out / 'segmentation.png', out / 'table_only.png')
    sys.stdout.flush()
    os._exit(0)  # pupil_apriltags crashes when its detector is freed


if __name__ == '__main__':
    main()
