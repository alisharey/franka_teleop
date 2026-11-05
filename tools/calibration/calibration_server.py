#!/usr/bin/env python3
"""Camera calibration page: hand-guide the arm, take snapshots, calibrate.

Put the robot in Programming mode in Desk (FCI active), guide the arm by hand
to a pose where the wrist camera sees the AprilTag on the table, let go, click
Snapshot. Repeat 15-25 times with different gripper tilts, then Calibrate.

Nothing here commands the robot: the robot state comes from robot_state_stream
(libfranka robot.read, read-only), the cameras are opened directly with
pyrealsense2 (so ./record.sh and realsense-viewer must not be running).

Each snapshot saves, in ~/franka_calibration/<session>/snapshots/<n>/:
    <cam>_color.png, <cam>_depth.png (16-bit, mm, aligned to color), snapshot.json
    (O_T_EE, q, the tag pose seen by each camera averaged over several frames,
    intrinsics, depth scale).
Calibrate writes calibration.json and check images to the session folder and
copies calibration.json to config/camera_calibration.json.

    tools/calibration/calibration_server.sh        # then open http://localhost:8770
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from calibrate_cameras import (HAND_EYE_METHODS, TagDetector, angle_deg, draw_axes, hand_eye,  # noqa: E402
                               inverse, mean_pose)

STILL_RAD_S = 0.01
FRAMES_PER_SNAPSHOT = 10


def read_camera_config(path):
    """cam name -> serial from config/recording.yaml (cameras: - name / serial)."""
    names = re.findall(r'-\s*name:\s*([\w-]+)\s*\n\s*serial:\s*"?(\d+)"?', Path(path).read_text())
    return dict(names)


def read_robot_ip(path):
    match = re.search(r'^robot_ip\s*=\s*([\d.]+)', Path(path).read_text(), re.M)
    return match.group(1) if match else None


class Camera(threading.Thread):
    def __init__(self, name, serial, width=640, height=480, fps=30):
        super().__init__(daemon=True)
        self.name, self.serial = name, serial
        self.size, self.fps = (width, height), fps
        self.lock = threading.Condition()
        self.color = self.depth = None
        self.frame_id = 0
        self.stamp = 0.0
        self.K = None
        self.depth_scale = None
        self.error = None
        self.rate = 0.0

    def run(self):
        import pyrealsense2 as rs
        while True:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(self.serial)
                config.enable_stream(rs.stream.color, self.size[0], self.size[1], rs.format.bgr8, self.fps)
                config.enable_stream(rs.stream.depth, self.size[0], self.size[1], rs.format.z16, self.fps)
                profile = pipeline.start(config)
                align = rs.align(rs.stream.color)
                intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
                self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                self.error = None
                times = []
                while True:
                    frames = align.process(pipeline.wait_for_frames(2000))
                    color, depth = frames.get_color_frame(), frames.get_depth_frame()
                    if not color or not depth:
                        continue
                    now = time.monotonic()
                    times = [t for t in times if now - t < 1.0] + [now]
                    with self.lock:
                        self.color = np.asanyarray(color.get_data()).copy()
                        self.depth = np.asanyarray(depth.get_data()).copy()
                        self.frame_id += 1
                        self.stamp = now
                        self.rate = len(times)
                        self.lock.notify_all()
            except Exception as error:  # camera busy or unplugged: retry
                self.error = str(error)
                try:
                    pipeline.stop()
                except Exception:
                    pass
                time.sleep(2.0)

    def latest(self):
        with self.lock:
            return self.color, self.depth, self.frame_id, self.stamp

    def next_frames(self, count, timeout=3.0):
        """`count` new consecutive frames (color, depth)."""
        out = []
        deadline = time.monotonic() + timeout
        with self.lock:
            seen = self.frame_id
            while len(out) < count and time.monotonic() < deadline:
                self.lock.wait(0.2)
                if self.frame_id != seen:
                    seen = self.frame_id
                    out.append((self.color.copy(), self.depth.copy()))
        return out


class Robot(threading.Thread):
    """robot_state_stream output (read-only)."""

    def __init__(self, ip, binary):
        super().__init__(daemon=True)
        self.ip, self.binary = ip, binary
        self.lock = threading.Lock()
        self.state = None
        self.received = 0.0
        self.error = None
        self.history = []  # (time, dq_max)

    def run(self):
        while True:
            # the FCI takes one client: connecting would knock a running teleop off the robot
            if subprocess.run(['pgrep', '-x', 'gamepad_teleop'], stdout=subprocess.DEVNULL).returncode == 0:
                self.error = 'the teleop is connected to the robot; stop it to take snapshots'
                time.sleep(2.0)
                continue
            try:
                process = subprocess.Popen([self.binary, self.ip, '20'], stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if 'error' in sample:
                        self.error = sample['error']
                        continue
                    now = time.monotonic()
                    with self.lock:
                        self.state = sample
                        self.received = now
                        self.error = None
                        self.history = [h for h in self.history if now - h[0] < 1.0] + [(now, sample['dq_max'])]
                process.wait()
            except Exception as error:
                self.error = str(error)
            time.sleep(2.0)

    def snapshot(self):
        with self.lock:
            if self.state is None or time.monotonic() - self.received > 1.0:
                return None
            still = len(self.history) >= 5 and max(h[1] for h in self.history) < STILL_RAD_S
            return dict(self.state, still=still)


class Session:
    def __init__(self, root, cameras, robot, detector, wrist, static):
        self.root = Path(root)
        (self.root / 'snapshots').mkdir(parents=True, exist_ok=True)
        self.cameras, self.robot, self.detector = cameras, robot, detector
        self.wrist, self.static = wrist, static
        self.detect_lock = threading.Lock()
        self.live = {}  # cam -> latest detection summary for the page
        self.busy = None
        self.last_calibration = None
        if (self.root / 'calibration.json').exists():
            self.last_calibration = json.loads((self.root / 'calibration.json').read_text())
        threading.Thread(target=self._live_detection, daemon=True).start()

    def detect(self, bgr, cam):
        with self.detect_lock:
            return self.detector.detect(bgr, self.cameras[cam].K)

    def _live_detection(self):
        while True:
            for name, camera in self.cameras.items():
                color, _, _, stamp = camera.latest()
                if color is None or camera.K is None:
                    self.live[name] = None
                    continue
                dets = self.detect(color, name)
                self.live[name] = [{'id': d['id'], 'distance_m': round(float(np.linalg.norm(d['pose'][:3, 3])), 3),
                                    'corners': d['corners'].tolist(), 'margin': round(d['margin'], 1)} for d in dets]
            time.sleep(0.2)

    # ------------------------------------------------------------------ snapshots
    def snapshots(self):
        out = []
        for folder in sorted((self.root / 'snapshots').iterdir(), key=lambda p: p.name):
            try:
                out.append(json.loads((folder / 'snapshot.json').read_text()))
            except (OSError, ValueError):
                continue
        return out

    def take_snapshot(self):
        state = self.robot.snapshot()
        if state is None:
            raise RuntimeError('no robot state: is FCI active and robot_state_stream connected?')
        if not state['still']:
            raise RuntimeError('the arm is moving: let go of it and wait a moment')
        existing = self.snapshots()
        index = max([s['index'] for s in existing], default=0) + 1
        folder = self.root / 'snapshots' / ('%03d' % index)
        folder.mkdir()
        record = {'index': index, 'time': datetime.now().isoformat(timespec='seconds'),
                  'robot_mode': state['mode'], 'q': state['q'],
                  'T_base_ee': np.array(state['O_T_EE']).reshape(4, 4, order='F').tolist(), 'cameras': {}}
        for name, camera in self.cameras.items():
            frames = camera.next_frames(FRAMES_PER_SNAPSHOT)
            if not frames:
                raise RuntimeError('camera %s delivers no frames' % name)
            cv2.imwrite(str(folder / ('%s_color.png' % name)), frames[0][0])
            cv2.imwrite(str(folder / ('%s_depth.png' % name)), frames[0][1])
            dets = [d for color, _ in frames for d in self.detect(color, name)]
            ids = [d['id'] for d in dets]
            entry = {'K': camera.K.tolist(), 'depth_scale': camera.depth_scale, 'frames_used': len(frames),
                     'width': frames[0][0].shape[1], 'height': frames[0][0].shape[0], 'tag': None}
            if ids:
                tag_id = max(set(ids), key=ids.count)
                mine = [d for d in dets if d['id'] == tag_id]
                entry['tag'] = {'id': tag_id, 'detections': len(mine),
                                'T_cam_tag': mean_pose([d['pose'] for d in mine]).tolist(),
                                'corners_first': mine[0]['corners'].tolist(),
                                'pose_first': mine[0]['pose'].tolist()}
            record['cameras'][name] = entry
        (folder / 'snapshot.json').write_text(json.dumps(record, indent=1))
        return record

    def delete_snapshot(self, index):
        folder = self.root / 'snapshots' / ('%03d' % int(index))
        if not folder.is_dir():
            raise RuntimeError('no snapshot %s' % index)
        shutil.rmtree(folder)

    def diversity(self, snapshots):
        """Gripper orientation span among snapshots with the tag in the wrist view."""
        rotations = [np.array(s['T_base_ee'])[:3, :3] for s in snapshots
                     if s['cameras'].get(self.wrist, {}).get('tag')]
        span = max([angle_deg(a.T @ b) for i, a in enumerate(rotations) for b in rotations[i + 1:]], default=0.0)
        nearest = None
        if len(rotations) >= 2:
            last = rotations[-1]
            nearest = min(angle_deg(last.T @ r) for r in rotations[:-1])
        return {'usable': len(rotations), 'span_deg': round(span, 1),
                'last_nearest_deg': None if nearest is None else round(nearest, 1)}

    # ------------------------------------------------------------------ calibration
    def calibrate(self, method='park'):
        snaps = self.snapshots()
        usable = [s for s in snaps if s['cameras'].get(self.wrist, {}).get('tag')]
        if len(usable) < 6:
            raise RuntimeError('only %d snapshots with the tag in the wrist view; take at least 6 (15-25 is better)'
                               % len(usable))
        ids = [s['cameras'][self.wrist]['tag']['id'] for s in usable]
        tag_id = max(set(ids), key=ids.count)
        usable = [s for s in usable if s['cameras'][self.wrist]['tag']['id'] == tag_id]
        ee = [np.array(s['T_base_ee']) for s in usable]
        cam_tag = [np.array(s['cameras'][self.wrist]['tag']['T_cam_tag']) for s in usable]
        t_ee_cam = hand_eye(ee, cam_tag, method)
        agreement = {}
        for name in HAND_EYE_METHODS:
            other = hand_eye(ee, cam_tag, name)
            agreement[name] = [round(float(np.linalg.norm(other[:3, 3] - t_ee_cam[:3, 3]) * 1000), 2),
                               round(angle_deg(t_ee_cam[:3, :3].T @ other[:3, :3]), 3)]
        tag_in_base = [e @ t_ee_cam @ c for e, c in zip(ee, cam_tag)]
        t_base_tag = mean_pose(tag_in_base)
        spread_mm = [float(np.linalg.norm(p[:3, 3] - t_base_tag[:3, 3]) * 1000) for p in tag_in_base]
        spread_deg = [angle_deg(t_base_tag[:3, :3].T @ p[:3, :3]) for p in tag_in_base]
        k_wrist = np.array(usable[0]['cameras'][self.wrist]['K'])
        reproj = [self.detector.reprojection_px(np.array(s['cameras'][self.wrist]['tag']['corners_first']),
                                                inverse(e @ t_ee_cam) @ t_base_tag,
                                                np.array(s['cameras'][self.wrist]['tag']['pose_first']), k_wrist)
                  for s, e in zip(usable, ee)]
        per_snapshot = [{'index': s['index'], 'tag_offset_mm': round(m, 1), 'tag_offset_deg': round(d, 2),
                         'reprojection_px': round(r, 2)} for s, m, d, r in zip(usable, spread_mm, spread_deg, reproj)]

        result = {'session': str(self.root), 'tag_id': tag_id, 'snapshots_used': len(usable),
                  'hand_eye_method': method, 'frames': {'base': 'robot base (libfranka O)',
                                                        'ee': 'libfranka EE frame (O_T_EE)',
                                                        'camera': 'optical: x right, y down, z forward'},
                  'T_ee_cam_' + self.wrist: t_ee_cam.tolist(), 'T_base_tag': t_base_tag.tolist(),
                  'intrinsics': {}, 'stats': {
                      'hand_eye_methods_vs_chosen_mm_deg': agreement,
                      'tag_spread_mm_median': float(np.median(spread_mm)), 'tag_spread_mm_max': float(np.max(spread_mm)),
                      'tag_spread_deg_median': float(np.median(spread_deg)),
                      'tag_spread_deg_max': float(np.max(spread_deg)),
                      'wrist_reprojection_px_median': float(np.median(reproj)),
                      'orientation_span_deg': self.diversity(usable)['span_deg']},
                  'per_snapshot': per_snapshot}
        for name in self.cameras:
            first = next((s['cameras'][name] for s in snaps if name in s['cameras']), None)
            if first:
                result['intrinsics'][name] = {'K': first['K'], 'width': first['width'], 'height': first['height'],
                                              'distortion': [0, 0, 0, 0, 0], 'depth_scale': first['depth_scale']}

        static = [s for s in snaps if (s['cameras'].get(self.static, {}).get('tag') or {}).get('id') == tag_id]
        if static:
            poses = [np.array(s['cameras'][self.static]['tag']['T_cam_tag']) for s in static]
            cam_tag_static = mean_pose(poses)
            t_base_static = t_base_tag @ inverse(cam_tag_static)
            result['T_base_cam_' + self.static] = t_base_static.tolist()
            result['stats'].update({
                'static_snapshots': len(static),
                'static_tag_distance_m': float(np.linalg.norm(cam_tag_static[:3, 3])),
                'static_tag_spread_mm_median': float(np.median(
                    [np.linalg.norm(p[:3, 3] - cam_tag_static[:3, 3]) * 1000 for p in poses])),
                'static_tag_spread_deg_median': float(np.median(
                    [angle_deg(cam_tag_static[:3, :3].T @ p[:3, :3]) for p in poses]))})
            k_static = np.array(static[0]['cameras'][self.static]['K'])
            checks = []
            for n, s in enumerate(snaps[:: max(1, len(snaps) // 4)][:4]):
                image = cv2.imread(str(self.root / 'snapshots' / ('%03d' % s['index']) / ('%s_color.png' % self.static)))
                if image is None:
                    continue
                cam_from_base = inverse(t_base_static)
                draw_axes(image, k_static, cam_from_base, 0.15, 'base')
                draw_axes(image, k_static, cam_from_base @ np.array(s['T_base_ee']), 0.08, 'EE (FK)')
                draw_axes(image, k_static, cam_from_base @ t_base_tag, 0.05, 'tag')
                path = 'check_%s_%d.png' % (self.static, n)
                cv2.imwrite(str(self.root / path), image)
                checks.append(path)
            result['check_images'] = checks
        else:
            result['warning'] = 'the %s camera did not see tag %d in any snapshot' % (self.static, tag_id)
        (self.root / 'calibration.json').write_text(json.dumps(result, indent=1))
        shutil.copy(self.root / 'calibration.json', ROOT / 'config' / 'camera_calibration.json')
        self.last_calibration = result
        return result

    # ------------------------------------------------------------------ page data
    def status(self):
        robot = self.robot.snapshot()
        robot_out = {'connected': robot is not None, 'error': self.robot.error}
        if robot:
            pose = np.array(robot['O_T_EE']).reshape(4, 4, order='F')
            robot_out.update(mode=robot['mode'], still=robot['still'], ee_xyz=[round(v, 4) for v in pose[:3, 3]],
                             dq_max=round(robot['dq_max'], 4))
        cams = {}
        for name, camera in self.cameras.items():
            live = self.live.get(name)
            cams[name] = {'fps': camera.rate, 'error': camera.error,
                          'tags': [{k: v for k, v in d.items() if k != 'corners'} for d in (live or [])]}
        snaps = self.snapshots()
        listing = [{'index': s['index'], 'time': s['time'],
                    'ee_xyz': [round(v, 3) for v in np.array(s['T_base_ee'])[:3, 3]],
                    'wrist_tag': bool(s['cameras'].get(self.wrist, {}).get('tag')),
                    'static_tag': bool(s['cameras'].get(self.static, {}).get('tag'))} for s in snaps]
        return {'ok': True, 'session': str(self.root), 'robot': robot_out, 'cameras': cams, 'snapshots': listing,
                'diversity': self.diversity(snaps), 'busy': self.busy, 'wrist': self.wrist, 'static': self.static,
                'calibration': None if not self.last_calibration else {
                    k: self.last_calibration.get(k) for k in ('snapshots_used', 'stats', 'per_snapshot', 'check_images',
                                                              'warning', 'tag_id')}}


def mjpeg_frame(session, name):
    color, _, _, _ = session.cameras[name].latest()
    if color is None:
        image = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(image, 'no frames: %s' % (session.cameras[name].error or 'starting'), (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        image = color.copy()
        for det in session.live.get(name) or []:
            corners = np.array(det['corners'], np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 0), 2)
            cv2.putText(image, 'tag %d  %.2f m' % (det['id'], det['distance_m']), tuple(corners[0] + [0, -8]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


def make_handler(session, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ('/', '/index.html'):
                body = page.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if path == '/api/status':
                return self._json(session.status())
            if path.startswith('/stream/') and path[8:] in session.cameras:
                name = path[8:]
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        jpeg = mjpeg_frame(session, name)
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpeg))
                        self.wfile.write(jpeg + b'\r\n')
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    return
            if path.startswith('/files/'):
                target = (session.root / path[7:]).resolve()
                if session.root.resolve() not in target.parents or not target.is_file():
                    return self._json({'ok': False, 'error': 'not found'}, 404)
                body = target.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png' if target.suffix == '.png' else 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self._json({'ok': False, 'error': 'not found'}, 404)

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            try:
                length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(length) or b'{}')
                if session.busy:
                    raise RuntimeError('busy: %s' % session.busy)
                if path == '/api/snapshot':
                    session.busy = 'taking a snapshot'
                    try:
                        record = session.take_snapshot()
                    finally:
                        session.busy = None
                    return self._json({'ok': True, 'index': record['index'],
                                       'wrist_tag': bool(record['cameras'][session.wrist]['tag']),
                                       'static_tag': bool(record['cameras'][session.static]['tag'])})
                if path.startswith('/api/snapshot/') and path.endswith('/delete'):
                    session.delete_snapshot(path.split('/')[3])
                    return self._json({'ok': True})
                if path == '/api/calibrate':
                    session.busy = 'calibrating'
                    try:
                        result = session.calibrate(body.get('method', 'park'))
                    finally:
                        session.busy = None
                    return self._json({'ok': True, 'stats': result['stats']})
                return self._json({'ok': False, 'error': 'not found'}, 404)
            except Exception as error:
                return self._json({'ok': False, 'error': str(error)}, 409)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--session', default=None, help='continue this session folder (default: a new one)')
    parser.add_argument('--root', default=str(Path.home() / 'franka_calibration'))
    parser.add_argument('--wrist', default='cam_1')
    parser.add_argument('--static', default='cam_2')
    parser.add_argument('--family', default='tagStandard41h12')
    parser.add_argument('--tag-size', type=float, default=0.0556, help='black-square edge [m]')
    args = parser.parse_args()

    serials = read_camera_config(ROOT / 'config' / 'recording.yaml')
    robot_ip = read_robot_ip(ROOT / 'config' / 'teleop.conf')
    binary = ROOT / 'install' / 'franka_teleop' / 'lib' / 'franka_teleop' / 'robot_state_stream'
    if not binary.exists():
        sys.exit('build first: colcon build (robot_state_stream missing)')
    cameras = {name: Camera(name, serials[name]) for name in (args.wrist, args.static)}
    for camera in cameras.values():
        camera.start()
    robot = Robot(robot_ip, str(binary))
    robot.start()
    root = Path(args.session) if args.session else Path(args.root) / datetime.now().strftime('%Y%m%d_%H%M%S')
    session = Session(root, cameras, robot, TagDetector(args.family, args.tag_size, -1), args.wrist, args.static)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(session, HERE / 'web' / 'index.html'))
    server.daemon_threads = True
    print('Calibration page: http://localhost:%d   (session %s, robot %s, cameras %s)' % (
        args.port, root, robot_ip, {n: c.serial for n, c in cameras.items()}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    sys.stdout.flush()
    os._exit(0)  # pupil_apriltags crashes when its detector is freed


if __name__ == '__main__':
    main()
