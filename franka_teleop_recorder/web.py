"""Web UI: live camera views, record control, episode list and replay.

Plain standard-library HTTP server (no extra Python packages). Endpoints:
  GET  /                               the page (web/index.html)
  GET  /stream/<camera>                live MJPEG of the color stream
  GET  /api/status                     recorder, cameras, teleop, disk
  POST /api/record  {"action": "start"|"stop"|"toggle"}
  POST /api/settings {"task": ..., "next_name": ...}
  GET  /api/episodes
  POST /api/episodes/<id>  {"name", "task", "notes", "success": true|false|null}
  POST /api/episodes/<id>/trash        moves the folder to <data_dir>/.trash
  GET  /episodes/<id>/<file>           episode files (replay.mp4 etc.), with Range support
"""

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlparse

CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.mp4': 'video/mp4',
    '.mkv': 'video/x-matroska',
    '.csv': 'text/csv; charset=utf-8',
    '.json': 'application/json',
}
STREAM_PERIOD_S = 1.0 / 15.0
_RANGE = re.compile(r'bytes=(\d*)-(\d*)$')


def make_server(recorder, store, web_dir, host, port):
    web_dir = Path(web_dir)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):  # keep the terminal quiet
            pass

        # ------------------------------------------------------------ helpers
        def _json(self, data, status=HTTPStatus.OK):
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._json({'ok': False, 'error': message}, status)

        def _body(self):
            length = int(self.headers.get('Content-Length') or 0)
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length) or b'{}')

        def _file(self, path):
            path = Path(path)
            if not path.is_file():
                return self._error(HTTPStatus.NOT_FOUND, 'not found')
            size = path.stat().st_size
            start, end = 0, size - 1
            status = HTTPStatus.OK
            match = _RANGE.match(self.headers.get('Range', ''))
            if match and size > 0:
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(match.group(2) or 0))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header('Content-Range', 'bytes */%d' % size)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            self.send_response(status)
            self.send_header('Content-Type', CONTENT_TYPES.get(path.suffix, 'application/octet-stream'))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Cache-Control', 'no-cache')
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
            self.end_headers()
            with open(path, 'rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = stream.read(min(1 << 18, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _stream(self, camera):
            feed = recorder.cameras.get(camera)
            if feed is None:
                return self._error(HTTPStatus.NOT_FOUND, 'unknown camera')
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            last_id = None
            try:
                while True:
                    frame_id, jpeg = feed.jpeg()
                    if jpeg is not None and frame_id != last_id:
                        last_id = frame_id
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n'
                                         % len(jpeg))
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    time.sleep(STREAM_PERIOD_S)
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

        # ------------------------------------------------------------ routes
        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ('/', '/index.html'):
                return self._file(web_dir / 'index.html')
            if path == '/api/status':
                return self._json(recorder.status())
            if path == '/api/episodes':
                return self._json({'episodes': store.list(), 'version': store.version})
            if path.startswith('/stream/'):
                return self._stream(path[len('/stream/'):])
            parts = path.strip('/').split('/')
            if len(parts) == 3 and parts[0] == 'episodes':
                try:
                    folder = store.path(parts[1])
                except KeyError:
                    return self._error(HTTPStatus.NOT_FOUND, 'unknown episode')
                target = (folder / parts[2]).resolve()
                if target.parent != folder.resolve():
                    return self._error(HTTPStatus.FORBIDDEN, 'forbidden')
                return self._file(target)
            return self._error(HTTPStatus.NOT_FOUND, 'not found')

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            try:
                body = self._body()
            except ValueError:
                return self._error(HTTPStatus.BAD_REQUEST, 'invalid JSON')
            if path == '/api/record':
                action = body.get('action', 'toggle')
                handler = {'start': recorder.start, 'stop': recorder.stop, 'toggle': recorder.toggle}.get(action)
                if handler is None:
                    return self._error(HTTPStatus.BAD_REQUEST, 'action must be start, stop or toggle')
                return self._json(handler())
            if path == '/api/settings':
                if 'task' in body:
                    recorder.task = str(body['task']).strip()
                if 'next_name' in body:
                    recorder.next_name = str(body['next_name']).strip()
                return self._json({'ok': True})
            parts = path.strip('/').split('/')
            if len(parts) in (3, 4) and parts[:2] == ['api', 'episodes']:
                episode_id = parts[2]
                try:
                    if len(parts) == 4 and parts[3] == 'trash':
                        store.move_to_trash(episode_id)
                        return self._json({'ok': True})
                    if len(parts) == 3:
                        new_id = store.update(
                            episode_id,
                            name=body.get('name'),
                            task=body.get('task'),
                            notes=body.get('notes'),
                            success=body.get('success') if 'success' in body else None,
                            clear_success='success' in body and body['success'] is None)
                        return self._json({'ok': True, 'id': new_id})
                except KeyError:
                    return self._error(HTTPStatus.NOT_FOUND, 'unknown episode')
                except (RuntimeError, FileExistsError) as exc:
                    return self._error(HTTPStatus.CONFLICT, str(exc))
            return self._error(HTTPStatus.NOT_FOUND, 'not found')

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
