#!/usr/bin/env python3
"""Step 2 of calibrate.sh: camera poses from one recorded episode and one AprilTag.

Record an episode (the normal recorder) in which the arm stops at 15-25 poses
where the wrist camera sees the tag on the table, with clearly different
orientations (tilt 20-40 deg about different axes), and the third-person camera
sees the tag too. Then:

  wrist camera   eye-in-hand calibration (cv2.calibrateHandEye): the tag is fixed, so
                 T_base_ee(i) . X . T_cam_tag(i) is the same for every pose i;
                 X = T_ee_cam is the camera's pose on the gripper, and the
                 tag's pose in the base frame comes out too
  static camera  T_base_cam = T_base_tag . T_cam_tag^-1, averaged over frames

Only frames where the arm is still are used, so camera/robot clock offsets do
not matter. Writes <episode>/calibration/calibration.json, check images that
draw the robot's end effector and base axes into the static camera's view using
the result, and prints the residuals:

  tag spread       where each wrist pose puts the tag in the base frame [mm, deg]
  reprojection     tag corners predicted from forward kinematics vs detected [px]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------- SE(3) helpers
def make_pose(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.ravel(translation)
    return pose


def inverse(pose):
    rotation, translation = pose[:3, :3], pose[:3, 3]
    return make_pose(rotation.T, -rotation.T @ translation)


def rotvec(rotation):
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    if np.pi - angle < 1e-6:  # near 180 deg: axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1) / 2, 0))
        largest = int(np.argmax(axis))
        for k in range(3):
            if k != largest:
                axis[k] = np.copysign(axis[k], rotation[largest, k] + rotation[k, largest])
        return angle * axis / np.linalg.norm(axis)
    return angle * axis / (2 * np.sin(angle))


def angle_deg(rotation):
    return float(np.degrees(np.linalg.norm(rotvec(rotation))))


def mean_rotation(rotations):
    """Chordal L2 mean (projects the summed matrices back onto SO(3))."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def mean_pose(poses):
    poses = np.asarray(poses)
    return make_pose(mean_rotation(poses[:, :3, :3]), poses[:, :3, 3].mean(axis=0))


HAND_EYE_METHODS = {'park': cv2.CALIB_HAND_EYE_PARK, 'tsai': cv2.CALIB_HAND_EYE_TSAI,
                    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS, 'horaud': cv2.CALIB_HAND_EYE_HORAUD}


def hand_eye(ee_poses, cam_tag_poses, method='park'):
    """X = T_ee_cam from cv2.calibrateHandEye (eye-in-hand): gripper->base = T_base_ee,
    target->cam = T_cam_tag. (OpenCV 4.x; the 5.0.0 Python package lacks it.)"""
    rotation, translation = cv2.calibrateHandEye(
        [p[:3, :3] for p in ee_poses], [p[:3, 3] for p in ee_poses],
        [p[:3, :3] for p in cam_tag_poses], [p[:3, 3] for p in cam_tag_poses], method=HAND_EYE_METHODS[method])
    return make_pose(rotation, translation)


# --------------------------------------------------------------------- data
def read_frames_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r['frame']) for r in rows]), np.array([int(r['stamp_ns']) for r in rows], dtype=np.int64)


_KEEP_ALIVE = []  # pupil_apriltags 1.0.4 crashes when a Detector is freed: never free it


class TagDetector:
    def __init__(self, family, tag_size, tag_id):
        import pupil_apriltags
        self.detector = pupil_apriltags.Detector(families=family, nthreads=4, quad_decimate=1.0,
                                                 refine_edges=1, decode_sharpening=0.25)
        _KEEP_ALIVE.append(self.detector)
        self.tag_size = tag_size
        self.tag_id = tag_id
        half = tag_size / 2
        # tag frame (AprilTag/pupil convention): x right, y down, z into the tag
        self.corners_tag = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])

    def detect(self, bgr, camera_matrix):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
        found = []
        for det in self.detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy),
                                        tag_size=self.tag_size):
            if self.tag_id >= 0 and det.tag_id != self.tag_id:
                continue
            if det.decision_margin < 20 or det.pose_R is None:
                continue
            pose = make_pose(det.pose_R, det.pose_t)
            found.append({'id': int(det.tag_id), 'pose': pose, 'corners': np.asarray(det.corners, float),
                          'margin': float(det.decision_margin), 'pose_err': float(det.pose_err)})
        return found

    def project(self, pose, camera_matrix):
        points = (pose[:3, :3] @ self.corners_tag.T).T + pose[:3, 3]
        pixels = (camera_matrix @ points.T).T
        return pixels[:, :2] / pixels[:, 2:3]

    def reprojection_px(self, detected_corners, predicted_pose, own_pose, camera_matrix):
        """Mean corner error; corners are matched through the detection's own pose
        (so the corner order of the detector does not matter)."""
        own = self.project(own_pose, camera_matrix)
        order = [int(np.argmin(np.linalg.norm(detected_corners - p, axis=1))) for p in own]
        predicted = self.project(predicted_pose, camera_matrix)
        return float(np.mean(np.linalg.norm(detected_corners[order] - predicted, axis=1)))


def still_mask(robot, stamps, threshold, window_s):
    """True for camera stamps where the arm was still for +-window_s."""
    out = np.zeros(len(stamps), dtype=bool)
    half = int(window_s * 1e9)
    for k, stamp in enumerate(stamps):
        lo, hi = np.searchsorted(robot['stamp_ns'], [stamp - half, stamp + half])
        if hi - lo >= 3:
            out[k] = robot['dq_max'][lo:hi].max() < threshold
    return out


def ee_at(robot, stamp):
    k = int(np.clip(np.searchsorted(robot['stamp_ns'], stamp), 0, len(robot['stamp_ns']) - 1))
    return robot['o_t_ee'][k]


def detect_in_video(path, wanted, detector, camera_matrix, keep_images=()):
    """Detections for the frame indices in `wanted` (a set); also returns the
    images of the frames in keep_images."""
    capture = cv2.VideoCapture(str(path))
    last = max(wanted | set(keep_images)) if wanted or keep_images else -1
    found, images = {}, {}
    index = 0
    while index <= last:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            found[index] = detector.detect(bgr, camera_matrix)
        if index in keep_images:
            images[index] = bgr
        index += 1
    capture.release()
    return found, images


def group_poses(samples, max_mm, max_deg):
    """Consecutive still samples at the same end-effector pose -> one group."""
    groups = []
    for sample in samples:
        if groups:
            ref = groups[-1][0]['ee']
            delta = inverse(ref) @ sample['ee']
            if np.linalg.norm(delta[:3, 3]) * 1000 < max_mm and angle_deg(delta[:3, :3]) < max_deg:
                groups[-1].append(sample)
                continue
        groups.append([sample])
    return groups


def draw_axes(image, camera_matrix, pose_in_cam, length, label):
    if pose_in_cam[2, 3] <= 0.05:
        return
    rvec, _ = cv2.Rodrigues(pose_in_cam[:3, :3])
    cv2.drawFrameAxes(image, camera_matrix, np.zeros(5), rvec, pose_in_cam[:3, 3], length, 2)
    p = camera_matrix @ pose_in_cam[:3, 3]
    cv2.putText(image, label, (int(p[0] / p[2]) + 6, int(p[1] / p[2]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------- main
def calibrate(args):
    episode = Path(args.episode)
    meta = json.loads((episode / 'meta.json').read_text())
    robot = dict(np.load(args.poses))
    detector = TagDetector(args.family, args.tag_size, args.tag_id)
    return _calibrate(args, episode, meta, robot, detector)


def _calibrate(args, episode, meta, robot, detector):
    out_dir = episode / 'calibration'
    out_dir.mkdir(exist_ok=True)
    cams = {}
    for cam in (args.wrist, args.static):
        info = meta['cameras'][cam]
        k = np.array(info['color_camera_info']['k']).reshape(3, 3)
        frames, stamps = read_frames_csv(episode / info['streams']['color']['frames_csv'])
        cams[cam] = {'K': k, 'frames': frames, 'stamps': stamps, 'video': episode / info['streams']['color']['file'],
                     'size': (info['streams']['color']['width'], info['streams']['color']['height'])}

    # wrist camera: detections in still frames, grouped by arm pose
    wrist = cams[args.wrist]
    still = still_mask(robot, wrist['stamps'], args.still_rad_s, args.still_window_s)
    still_frames = set(int(f) for f in wrist['frames'][still])
    print('%s: %d frames, %d with the arm still' % (args.wrist, len(wrist['frames']), len(still_frames)))
    found, _ = detect_in_video(wrist['video'], still_frames, detector, wrist['K'])
    ids = [d['id'] for dets in found.values() for d in dets]
    if not ids:
        sys.exit('the wrist camera never saw a %s tag while the arm was still' % args.family)
    tag_id = args.tag_id if args.tag_id >= 0 else max(set(ids), key=ids.count)
    samples = []
    for frame, stamp in zip(wrist['frames'], wrist['stamps']):
        for det in found.get(int(frame), []):
            if det['id'] == tag_id:
                samples.append({'frame': int(frame), 'stamp': int(stamp), 'ee': ee_at(robot, stamp), **det})
    groups = [g for g in group_poses(samples, 3.0, 1.0) if len(g) >= args.min_frames_per_pose]
    print('tag %d seen by the wrist camera in %d still frames -> %d distinct arm poses (>= %d frames each)' % (
        tag_id, len(samples), len(groups), args.min_frames_per_pose))
    if len(groups) < args.min_poses:
        sys.exit('need at least %d poses; record more stops with the tag in the wrist view' % args.min_poses)
    ee = [mean_pose([s['ee'] for s in g]) for g in groups]
    cam_tag = [mean_pose([s['pose'] for s in g]) for g in groups]
    rotation_span = max(angle_deg((inverse(ee[i]) @ ee[j])[:3, :3]) for i in range(len(ee)) for j in range(len(ee)))

    t_ee_cam = hand_eye(ee, cam_tag, args.method)
    # the other OpenCV methods should agree within a few mm / tenths of a degree
    agreement = {}
    for name in HAND_EYE_METHODS:
        other = hand_eye(ee, cam_tag, name)
        agreement[name] = [round(float(np.linalg.norm(other[:3, 3] - t_ee_cam[:3, 3]) * 1000), 2),
                           round(angle_deg(t_ee_cam[:3, :3].T @ other[:3, :3]), 3)]
    tag_in_base = [ee[k] @ t_ee_cam @ cam_tag[k] for k in range(len(groups))]
    t_base_tag = mean_pose(tag_in_base)
    spread_mm = [float(np.linalg.norm(p[:3, 3] - t_base_tag[:3, 3]) * 1000) for p in tag_in_base]
    spread_deg = [angle_deg((inverse(t_base_tag) @ p)[:3, :3]) for p in tag_in_base]
    wrist_reproj = []
    for g, e in zip(groups, ee):
        predicted = inverse(e @ t_ee_cam) @ t_base_tag
        wrist_reproj += [detector.reprojection_px(s['corners'], predicted, s['pose'], wrist['K']) for s in g]

    # static camera: every frame that sees the tag (the camera and the tag do not move)
    static = cams[args.static]
    every = set(int(f) for f in static['frames'][::args.static_stride])
    check_frames = [int(static['frames'][k]) for k in np.linspace(0, len(static['frames']) - 1, 4).astype(int)]
    found_static, images = detect_in_video(static['video'], every, detector, static['K'], keep_images=set(check_frames))
    static_dets = [d for dets in found_static.values() for d in dets if d['id'] == tag_id]
    if not static_dets:
        sys.exit('the %s camera never saw tag %d' % (args.static, tag_id))
    cam2_tag = mean_pose([d['pose'] for d in static_dets])
    static_spread_mm = float(np.median([np.linalg.norm(d['pose'][:3, 3] - cam2_tag[:3, 3]) * 1000 for d in static_dets]))
    static_spread_deg = float(np.median([angle_deg((inverse(cam2_tag) @ d['pose'])[:3, :3]) for d in static_dets]))
    t_base_cam2 = t_base_tag @ inverse(cam2_tag)
    static_reproj = [detector.reprojection_px(d['corners'], cam2_tag, d['pose'], static['K']) for d in static_dets]

    # check images: the end effector (forward kinematics) and the base frame drawn into the static view
    stamp_of = dict(zip(static['frames'].tolist(), static['stamps'].tolist()))
    for n, frame in enumerate(check_frames):
        image = images.get(frame)
        if image is None:
            continue
        cam_from_base = inverse(t_base_cam2)
        draw_axes(image, static['K'], cam_from_base, 0.15, 'base')
        draw_axes(image, static['K'], cam_from_base @ ee_at(robot, stamp_of[frame]), 0.08, 'EE (FK)')
        draw_axes(image, static['K'], cam_from_base @ t_base_tag, 0.05, 'tag')
        cv2.imwrite(str(out_dir / ('check_%s_%d.png' % (args.static, n))), image)

    result = {
        'episode': meta['id'], 'tag': {'family': args.family, 'id': tag_id, 'size_m': args.tag_size},
        'frames': {'base': 'robot base (libfranka O)', 'ee': 'libfranka EE frame (O_T_EE)',
                   'camera': 'optical: x right, y down, z forward'},
        'T_base_cam_' + args.static: t_base_cam2.tolist(),
        'T_ee_cam_' + args.wrist: t_ee_cam.tolist(),
        'T_base_tag': t_base_tag.tolist(),
        'intrinsics': {cam: {'K': c['K'].tolist(), 'width': c['size'][0], 'height': c['size'][1],
                             'distortion': [0, 0, 0, 0, 0]} for cam, c in cams.items()},
        'stats': {
            'wrist_poses': len(groups), 'wrist_rotation_span_deg': rotation_span, 'hand_eye_method': args.method,
            'hand_eye_methods_vs_chosen_mm_deg': agreement,
            'tag_spread_mm_median': float(np.median(spread_mm)), 'tag_spread_mm_max': float(np.max(spread_mm)),
            'tag_spread_deg_median': float(np.median(spread_deg)), 'tag_spread_deg_max': float(np.max(spread_deg)),
            'wrist_reprojection_px_median': float(np.median(wrist_reproj)),
            'static_detections': len(static_dets), 'static_tag_spread_mm_median': static_spread_mm,
            'static_tag_spread_deg_median': static_spread_deg,
            'static_reprojection_px_median': float(np.median(static_reproj)),
            'static_tag_distance_m': float(np.linalg.norm(cam2_tag[:3, 3])),
        },
    }
    (out_dir / 'calibration.json').write_text(json.dumps(result, indent=1))
    s = result['stats']
    print('\nwrist camera %s: %d poses, gripper rotation span %.0f deg, method %s' % (
        args.wrist, s['wrist_poses'], s['wrist_rotation_span_deg'], args.method))
    print('  other OpenCV methods differ from it by [mm, deg]: %s' % agreement)
    print('  tag position spread in the base frame: median %.1f mm, max %.1f mm; rotation median %.2f deg, max %.2f'
          % (s['tag_spread_mm_median'], s['tag_spread_mm_max'], s['tag_spread_deg_median'], s['tag_spread_deg_max']))
    print('  reprojection (tag corners from FK vs detected): median %.2f px' % s['wrist_reprojection_px_median'])
    print('  T_ee_cam: translation %s mm' % np.round(t_ee_cam[:3, 3] * 1000, 1))
    print('static camera %s: tag seen in %d frames at %.2f m; frame-to-frame spread %.1f mm / %.2f deg' % (
        args.static, s['static_detections'], s['static_tag_distance_m'], static_spread_mm, static_spread_deg))
    print('  camera position in the base frame: %s m' % np.round(t_base_cam2[:3, 3], 3))
    print('\nwrote %s and check images (base, end effector and tag axes drawn into %s)' % (
        out_dir / 'calibration.json', args.static))
    return result


def self_test():
    """Synthetic poses with a known answer, plus detection-like noise."""
    rng = np.random.default_rng(0)

    def random_rotation(scale):
        v = rng.normal(size=3)
        angle = scale * rng.uniform(0.3, 1.0)
        v = v / np.linalg.norm(v) * angle
        return cv2.Rodrigues(v)[0]
    x_true = make_pose(random_rotation(np.pi), [0.05, -0.03, 0.04])
    tag_true = make_pose(random_rotation(np.pi), [0.5, 0.1, 0.0])
    ee, cam = [], []
    for _ in range(20):
        e = make_pose(random_rotation(0.6) @ cv2.Rodrigues(np.array([np.pi, 0, 0]))[0],
                      [0.45, 0.05, 0.35] + rng.normal(scale=0.05, size=3))
        c = inverse(e @ x_true) @ tag_true
        noise = make_pose(random_rotation(np.radians(0.3)), rng.normal(scale=0.0005, size=3))
        ee.append(e)
        cam.append(c @ noise)
    for method in HAND_EYE_METHODS:
        x = hand_eye(ee, cam, method)
        err_mm = np.linalg.norm(x[:3, 3] - x_true[:3, 3]) * 1000
        err_deg = angle_deg(inverse(x_true)[:3, :3] @ x[:3, :3])
        print('self-test %-10s X error %.2f mm / %.3f deg (noise 0.5 mm, 0.3 deg per pose)' % (method, err_mm, err_deg))
        assert err_mm < 3 and err_deg < 0.5, 'hand-eye calibration failed the synthetic test'
    print('self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', nargs='?', help='recorded episode folder')
    parser.add_argument('--poses', help='robot poses from extract_robot_poses.py')
    parser.add_argument('--wrist', default='cam_1', help='camera on the gripper')
    parser.add_argument('--static', default='cam_2', help='fixed third-person camera')
    parser.add_argument('--family', default='tagStandard41h12')
    parser.add_argument('--tag-size', type=float, default=0.0556, help='black-square edge [m] (as apriltag_pose.py)')
    parser.add_argument('--tag-id', type=int, default=-1, help='tag id (default: the one the wrist camera saw most)')
    parser.add_argument('--method', default='park', choices=sorted(HAND_EYE_METHODS), help='cv2.calibrateHandEye method')
    parser.add_argument('--still-rad-s', type=float, default=0.01, help='max joint speed counted as still')
    parser.add_argument('--still-window-s', type=float, default=0.3, help='still for this long before and after')
    parser.add_argument('--min-frames-per-pose', type=int, default=3)
    parser.add_argument('--min-poses', type=int, default=8)
    parser.add_argument('--static-stride', type=int, default=3, help='use every n-th static camera frame')
    parser.add_argument('--self-test', action='store_true', help='check the solver on synthetic data and exit')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.episode or not args.poses:
        parser.error('episode and --poses are required (or --self-test)')
    # pupil_apriltags 1.0.4 crashes in apriltag_detector_destroy when a detector is
    # freed, so end the process without running destructors (output flushed first).
    status = 0
    try:
        calibrate(args)
    except SystemExit as stop:
        if stop.code not in (None, 0):
            sys.stdout.flush()
            print(stop.code if isinstance(stop.code, str) else 'failed', file=sys.stderr)
            status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == '__main__':
    main()
