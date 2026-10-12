"""Camera re-placement against a fixed AprilTag, with a local web page.

Save the camera's pose relative to the tag as a named reference. Later, with a
reference selected, the page tells you how to move and turn the camera to get
back to it (in the camera's own directions: left/right, up/down,
closer/farther, tilt, pan, roll), draws where the tag sat in the saved view
(ghost outline), and can overlay the saved image.

Poses are averaged over the last --average frames. References live in
references/ next to this file (references.json plus one image per reference).
"""

import argparse
import datetime
import json
import re
import threading
import time
import uuid
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pupil_apriltags
import pyrealsense2 as rs

from apriltag_pose import euler_xyz_degrees, start_camera

HERE = Path(__file__).resolve().parent
REFERENCES_DIR = HERE / 'references'
REFERENCES_FILE = REFERENCES_DIR / 'references.json'
_ID = re.compile(r'^[0-9a-f]{12}$')


# ------------------------------------------------------------------ geometry

def rotation_vector(rotation):
    vector, _ = cv2.Rodrigues(rotation)
    return vector.reshape(3)


def average_transform(transforms):
    """Mean translation; rotation = nearest rotation to the mean matrix."""
    stack = np.array(transforms)
    u, _, vt = np.linalg.svd(stack[:, :3, :3].mean(axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = stack[:, :3, 3].mean(axis=0)
    return result


def correction(t_ref_cur, position_tolerance_m, angle_tolerance_deg):
    """What to do to bring the camera from its current pose back to the reference.

    t_ref_cur is the current camera pose in the reference camera frame (optical:
    x right, y down, z forward). The move is -translation; the turn is -rotation
    vector, expressed about the camera's own axes (nearly the same frames once close).
    """
    move = -t_ref_cur[:3, 3]
    turn = np.degrees(-rotation_vector(t_ref_cur[:3, :3]))
    steps = []
    for value, positive, negative in ((move[0], 'Move right', 'Move left'),
                                      (move[1], 'Move down', 'Move up'),
                                      (move[2], 'Move closer (forward)', 'Move back')):
        if abs(value) > position_tolerance_m:
            steps.append({'kind': 'move', 'text': '%s %.1f cm' % (positive if value > 0 else negative, abs(value) * 100),
                          'size': abs(value) / position_tolerance_m})
    for value, positive, negative in ((turn[0], 'Tilt up', 'Tilt down'),
                                      (turn[1], 'Pan right', 'Pan left'),
                                      (turn[2], 'Roll clockwise', 'Roll counter-clockwise')):
        if abs(value) > angle_tolerance_deg:
            steps.append({'kind': 'turn', 'text': '%s %.1f°' % (positive if value > 0 else negative, abs(value)),
                          'size': abs(value) / angle_tolerance_deg})
    steps.sort(key=lambda step: -step['size'])
    return {
        'aligned': not steps,
        'steps': [step['text'] for step in steps],
        'offset_mm': (t_ref_cur[:3, 3] * 1000).round(1).tolist(),
        'distance_mm': round(float(np.linalg.norm(t_ref_cur[:3, 3]) * 1000), 1),
        'angle_deg': round(float(np.degrees(np.linalg.norm(rotation_vector(t_ref_cur[:3, :3])))), 2),
        'tilt_pan_roll_deg': np.degrees(rotation_vector(t_ref_cur[:3, :3])).round(2).tolist(),
    }


# ------------------------------------------------------------------ references

class ReferenceStore:
    def __init__(self):
        REFERENCES_DIR.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self.items = json.loads(REFERENCES_FILE.read_text()) if REFERENCES_FILE.exists() else []
        self.active = self.items[-1]['id'] if self.items else None

    def _write(self):
        temporary = REFERENCES_FILE.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.items, indent=2))
        temporary.replace(REFERENCES_FILE)

    def get(self, reference_id):
        return next((item for item in self.items if item['id'] == reference_id), None)

    def add(self, name, t_tag_camera, image, info):
        with self._lock:
            reference_id = uuid.uuid4().hex[:12]
            cv2.imwrite(str(REFERENCES_DIR / ('%s.jpg' % reference_id)), image)
            item = dict(info, id=reference_id, name=name or 'reference %d' % (len(self.items) + 1),
                        created=datetime.datetime.now().isoformat(timespec='seconds'),
                        T_tag_camera=t_tag_camera.round(8).tolist())
            self.items.append(item)
            self.active = reference_id
            self._write()
            return item

    def delete(self, reference_id):
        with self._lock:
            self.items = [item for item in self.items if item['id'] != reference_id]
            (REFERENCES_DIR / ('%s.jpg' % reference_id)).unlink(missing_ok=True)
            if self.active == reference_id:
                self.active = self.items[-1]['id'] if self.items else None
            self._write()

    def rename(self, reference_id, name):
        with self._lock:
            item = self.get(reference_id)
            if item is None:
                raise KeyError(reference_id)
            item['name'] = name
            self._write()


# ------------------------------------------------------------------ camera loop

class Tracker(threading.Thread):
    def __init__(self, args, store):
        super().__init__(daemon=True)
        self.args = args
        self.store = store
        self.pipeline, self.camera_matrix, self.distortion, _, self.camera_name = start_camera(
            args.serial, args.width, args.height, args.fps, False)
        self.serial = self.camera_name.split()[-1]
        self.detector = pupil_apriltags.Detector(families=args.family, nthreads=4, quad_decimate=1.0,
                                                 refine_edges=1, decode_sharpening=0.25)
        self.history = deque(maxlen=args.average)  # T_tag_camera per frame, None when the tag is not seen
        self.lock = threading.Lock()
        self.jpeg = (0, None)
        self.clean_image = None
        self.status = {}
        self.stopping = False

    def camera_params(self):
        k = self.camera_matrix
        return k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    def tag_corners(self):
        half = self.args.tag_size / 2
        return np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])

    def project(self, t_camera_tag, points):
        rvec, _ = cv2.Rodrigues(t_camera_tag[:3, :3])
        projected, _ = cv2.projectPoints(points, rvec, t_camera_tag[:3, 3], self.camera_matrix, self.distortion)
        return projected.reshape(-1, 2).astype(int)

    def run(self):
        frame_times = deque(maxlen=30)
        while not self.stopping:
            frames = self.pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data()).copy()
            clean = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = [d for d in self.detector.detect(gray, True, self.camera_params(), self.args.tag_size)
                          if d.tag_id == self.args.tag_id]
            t_tag_camera = None
            if detections:
                detection = detections[0]
                t_camera_tag = np.eye(4)
                t_camera_tag[:3, :3] = detection.pose_R
                t_camera_tag[:3, 3] = detection.pose_t.reshape(3)
                t_tag_camera = np.linalg.inv(t_camera_tag)
            self.history.append(t_tag_camera)
            seen = [t for t in self.history if t is not None]
            averaged = average_transform(seen) if len(seen) >= max(3, len(self.history) // 2) and \
                t_tag_camera is not None else None

            reference = self.store.get(self.store.active) if self.store.active else None
            alignment = None
            if reference is not None:
                t_tag_reference = np.array(reference['T_tag_camera'])
                # Ghost: where the tag appeared in the saved view.
                ghost = self.project(np.linalg.inv(t_tag_reference), self.tag_corners())
                cv2.polylines(image, [ghost.reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)
                for corner in ghost:
                    cv2.circle(image, tuple(int(v) for v in corner), 4, (255, 0, 255), -1)
                if averaged is not None:
                    alignment = correction(np.linalg.inv(t_tag_reference) @ averaged,
                                           self.args.position_tolerance, self.args.angle_tolerance)
            if detections:
                detection = detections[0]
                cv2.polylines(image, [detection.corners.astype(int).reshape(-1, 1, 2)], True, (0, 255, 255), 2,
                              cv2.LINE_AA)
                rvec, _ = cv2.Rodrigues(detection.pose_R)
                cv2.drawFrameAxes(image, self.camera_matrix, self.distortion, rvec, detection.pose_t.reshape(3),
                                  self.args.tag_size * 0.6, 2)
            if alignment is not None:
                banner = 'ALIGNED' if alignment['aligned'] else alignment['steps'][0]
                colour = (0, 200, 0) if alignment['aligned'] else (0, 140, 255)
                cv2.rectangle(image, (0, image.shape[0] - 40), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
                cv2.putText(image, banner, (12, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2,
                            cv2.LINE_AA)

            now = time.time()
            frame_times.append(now)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            camera_in_tag = None
            if averaged is not None:
                camera_in_tag = {'position_mm': (averaged[:3, 3] * 1000).round(1).tolist(),
                                 'roll_pitch_yaw_deg': euler_xyz_degrees(averaged[:3, :3]).round(2).tolist()}
            with self.lock:
                if ok:
                    self.jpeg = (self.jpeg[0] + 1, encoded.tobytes())
                self.clean_image = clean
                self.status = {
                    'camera': self.camera_name,
                    'serial': self.serial,
                    'fps': round(fps, 1),
                    'tag_visible': bool(detections),
                    'tag_id': self.args.tag_id,
                    'tag_size_mm': self.args.tag_size * 1000,
                    'frames_with_tag': len(seen),
                    'average_window': len(self.history),
                    'camera_in_tag': camera_in_tag,
                    'tag_distance_mm': round(float(np.linalg.norm(averaged[:3, 3]) * 1000), 1)
                    if averaged is not None else None,
                    'alignment': alignment,
                    'active_reference': self.store.active,
                    'position_tolerance_mm': self.args.position_tolerance * 1000,
                    'angle_tolerance_deg': self.args.angle_tolerance,
                }

    def save_reference(self, name):
        """Saves the average of the last frames; needs the tag in all of them."""
        frames = list(self.history)
        if len(frames) < self.history.maxlen or any(t is None for t in frames):
            raise RuntimeError('hold the camera still with the tag visible for a moment, then save again')
        with self.lock:
            image = self.clean_image.copy()
        return self.store.add(name, average_transform(frames), image, {
            'camera': self.camera_name, 'serial': self.serial, 'tag_family': self.args.family,
            'tag_id': self.args.tag_id, 'tag_size_m': self.args.tag_size,
            'camera_matrix': self.camera_matrix.round(4).tolist()})


# ------------------------------------------------------------------ web server

def make_server(tracker, store, host, port):
    page = (HERE / 'web' / 'align.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def _send(self, body, content_type, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=HTTPStatus.OK):
            self._send(json.dumps(data).encode(), 'application/json', status)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                return self._send(page, 'text/html; charset=utf-8')
            if path == '/api/status':
                with tracker.lock:
                    status = dict(tracker.status)
                return self._json(dict(status, references=store.items))
            if path.startswith('/references/') and path.endswith('.jpg'):
                reference_id = path[len('/references/'):-4]
                image = REFERENCES_DIR / ('%s.jpg' % reference_id)
                if _ID.match(reference_id) and image.is_file():
                    return self._send(image.read_bytes(), 'image/jpeg')
                return self._json({'error': 'not found'}, HTTPStatus.NOT_FOUND)
            if path == '/stream':
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.end_headers()
                last = None
                try:
                    while True:
                        with tracker.lock:
                            frame_id, jpeg = tracker.jpeg
                        if jpeg is not None and frame_id != last:
                            last = frame_id
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n'
                                             % len(jpeg) + jpeg + b'\r\n')
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            self._json({'error': 'not found'}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(length) or b'{}') if length else {}
            except ValueError:
                return self._json({'error': 'invalid JSON'}, HTTPStatus.BAD_REQUEST)
            parts = self.path.strip('/').split('/')
            try:
                if parts == ['api', 'references']:
                    item = tracker.save_reference(str(body.get('name', '')).strip())
                    return self._json({'ok': True, 'id': item['id']})
                if len(parts) == 4 and parts[:2] == ['api', 'references'] and _ID.match(parts[2]):
                    if store.get(parts[2]) is None:
                        return self._json({'error': 'unknown reference'}, HTTPStatus.NOT_FOUND)
                    if parts[3] == 'activate':
                        store.active = parts[2]
                    elif parts[3] == 'delete':
                        store.delete(parts[2])
                    elif parts[3] == 'rename' and str(body.get('name', '')).strip():
                        store.rename(parts[2], str(body['name']).strip())
                    else:
                        return self._json({'error': 'unknown action'}, HTTPStatus.BAD_REQUEST)
                    return self._json({'ok': True})
                if parts == ['api', 'deactivate']:
                    store.active = None
                    return self._json({'ok': True})
            except RuntimeError as exc:
                return self._json({'error': str(exc)}, HTTPStatus.CONFLICT)
            self._json({'error': 'not found'}, HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag-size', type=float, default=0.0556,
                        help='black-square edge in metres (default 0.0556: a 100 mm printed tagStandard41h12)')
    parser.add_argument('--tag-id', type=int, default=4)
    parser.add_argument('--family', default='tagStandard41h12')
    parser.add_argument('--serial', default='', help='RealSense serial number (default: the first camera)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--average', type=int, default=15, help='frames averaged for the pose (default 15)')
    parser.add_argument('--position-tolerance', type=float, default=0.002, help='metres (default 0.002)')
    parser.add_argument('--angle-tolerance', type=float, default=0.3, help='degrees (default 0.3)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()

    store = ReferenceStore()
    tracker = Tracker(args, store)
    try:
        server = make_server(tracker, store, args.host, args.port)
    except OSError as exc:
        tracker.pipeline.stop()
        raise SystemExit('cannot use port %d (%s); is camera_align already running? Use --port.' % (
            args.port, exc.strerror))
    tracker.start()
    print('%s; tag %s id %d (%.1f mm). Open http://localhost:%d  (Ctrl-C quits)'
          % (tracker.camera_name, args.family, args.tag_id, args.tag_size * 1000, args.port), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stopping = True
        tracker.join(timeout=2)
        tracker.pipeline.stop()
        server.server_close()


if __name__ == '__main__':
    main()
