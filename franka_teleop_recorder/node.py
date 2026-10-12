"""ROS 2 recorder node.

Always subscribed (for the live view and status):
  /cameras/<name>/color/image_raw, .../aligned_depth_to_color/image_raw (+ camera_info)
  /teleop/joy       record button (rising edge toggles recording)
  /teleop/session   teleop config text and git commit (latched)
  every other topic, discovered once a second, as raw serialized bytes

While recording, images go to per-camera video files (video.py) and every
non-image message is written unchanged into the episode's ROS bag, so the bag
holds everything published (1 kHz /teleop/state, /tf_static, camera_info, IMU, ...).
"""

import dataclasses
import json
import queue
import shutil
import threading
import time
from collections import deque

import cv2
import numpy as np
import rosbag2_py
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo, Image, Joy
from std_msgs.msg import String

from franka_teleop.msg import TeleopState

from .episodes import read_json, write_json
from .video import PIXEL_FORMATS, VideoWriter, color_command, depth_command, make_side_by_side

IMAGE_TYPES = {'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'}
SKIP_TOPICS = {'/parameter_events'}
STATE_TOPIC = '/teleop/state'
MIN_FREE_BYTES = 2e9
LATCHED_CACHE_SIZE = 200
ROBOT_MODES = ['other', 'idle', 'move', 'guiding', 'reflex', 'user_stopped', 'automatic_error_recovery']


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class RateMeter:
    """Messages per second over the last two seconds."""

    def __init__(self, window_s=2.0):
        self.window_ns = int(window_s * 1e9)
        self.times = deque()

    def tick(self, now_ns):
        self.times.append(now_ns)
        while self.times and now_ns - self.times[0] > self.window_ns:
            self.times.popleft()

    def rate(self):
        now_ns = time.time_ns()
        while self.times and now_ns - self.times[0] > self.window_ns:
            self.times.popleft()
        return len(self.times) / (self.window_ns / 1e9)


class CameraFeed:
    """Latest frame and stream statistics of one camera, for the live view."""

    def __init__(self, name, serial):
        self.name = name
        self.serial = serial
        self.color_rate = RateMeter()
        self.depth_rate = RateMeter()
        self.color_info = None
        self.depth_info = None
        self.width = 0
        self.height = 0
        self.encoding = ''
        self.last_color_ns = 0
        self._lock = threading.Lock()
        self._latest = None
        self._latest_id = 0
        self._jpeg = (0, None)

    def set_latest(self, msg):
        with self._lock:
            self._latest = msg
            self._latest_id += 1

    def jpeg(self):
        """(frame id, JPEG bytes) of the newest color frame, encoded once per frame."""
        with self._lock:
            msg, frame_id = self._latest, self._latest_id
            if msg is None or self._jpeg[0] == frame_id:
                return self._jpeg
        pixels = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        pixels = pixels[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._lock:
            if ok:
                self._jpeg = (frame_id, encoded.tobytes())
            return self._jpeg


class BagWriter:
    """Writes serialized messages into a ROS 2 bag from its own thread."""

    def __init__(self, uri, storage):
        self.storage = storage
        self.topics = {}
        self.error = None
        self._writer = rosbag2_py.SequentialWriter()
        try:
            self._open(uri, storage)
        except RuntimeError:
            if storage == 'sqlite3':
                raise
            # mcap plugin not installed: the failed attempt may have created the
            # (empty) bag folder already, and sqlite3 refuses to reuse it.
            shutil.rmtree(uri, ignore_errors=True)
            self.storage = 'sqlite3'
            self._writer = rosbag2_py.SequentialWriter()
            self._open(uri, 'sqlite3')
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self, uri, storage):
        options = rosbag2_py.StorageOptions(uri=str(uri), storage_id=storage)
        if storage == 'mcap' and hasattr(options, 'storage_preset_profile'):
            options.storage_preset_profile = 'zstd_fast'
        self._writer.open(options, rosbag2_py.ConverterOptions('cdr', 'cdr'))

    def write(self, topic, type_name, data, time_ns):
        self._queue.put((topic, type_name, data, time_ns))

    def backlog(self):
        return self._queue.qsize()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            topic, type_name, data, time_ns = item
            try:
                if topic not in self.topics:
                    self._writer.create_topic(rosbag2_py.TopicMetadata(
                        name=topic, type=type_name, serialization_format='cdr'))
                    self.topics[topic] = 0
                self._writer.write(topic, data, time_ns)
                self.topics[topic] += 1
            except Exception as exc:  # keep recording the other topics
                self.error = 'bag write failed on %s: %s' % (topic, exc)

    def close(self):
        self._queue.put(None)
        self._thread.join()
        if hasattr(self._writer, 'close'):
            self._writer.close()
        self._writer = None  # the bag is finalized when the writer is destroyed


class ActiveEpisode:
    def __init__(self, episode_id, path, meta, config, latched):
        self.id = episode_id
        self.path = path
        self.meta = meta
        self.config = config
        self.start_ns = time.time_ns()
        self.bag = BagWriter(path / 'bag', config.bag_storage)
        # Latched topics (e.g. /tf_static, /teleop/session) were published before
        # this episode began; store their last messages at their receive times.
        for topic, type_name, data, time_ns in latched:
            self.bag.write(topic, type_name, data, time_ns)
        self.writers = {}
        self._lock = threading.Lock()
        self._closed = False

    def write_frame(self, camera, stream, msg, frame_stamp_ns, receive_ns):
        key = (camera, stream)
        writer = self.writers.get(key)
        if writer is None:
            with self._lock:
                if self._closed:
                    return
                writer = self.writers.get(key) or self._open_writer(camera, stream, msg)
                if writer is None:
                    return
        if self._closed:
            return
        data = msg.data
        pixel_format, bytes_per_pixel = PIXEL_FORMATS[msg.encoding]
        if msg.step != msg.width * bytes_per_pixel:  # strip row padding
            rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            data = rows[:, :msg.width * bytes_per_pixel].tobytes()
        writer.put(data, frame_stamp_ns, receive_ns)

    def _open_writer(self, camera, stream, msg):
        if msg.encoding not in PIXEL_FORMATS:
            return None
        pixel_format, _ = PIXEL_FORMATS[msg.encoding]
        fps = self.config.fps
        if stream == 'color':
            command = color_command(self.path / ('%s_color.mp4' % camera), msg.width, msg.height, fps,
                                    pixel_format, self.config.color_crf, self.config.color_preset)
        else:
            command = depth_command(self.path / ('%s_depth.mkv' % camera), msg.width, msg.height, fps,
                                    pixel_format)
        writer = VideoWriter(command, self.path / ('%s_%s_frames.csv' % (camera, stream)))
        writer.width, writer.height, writer.encoding = msg.width, msg.height, msg.encoding
        self.writers[(camera, stream)] = writer
        return writer

    def close(self):
        with self._lock:
            self._closed = True
        errors = []
        for writer in self.writers.values():
            error = writer.close()
            if error:
                errors.append(error)
        self.bag.close()
        if self.bag.error:
            errors.append(self.bag.error)
        return errors


class Recorder(Node):
    def __init__(self, config, store):
        super().__init__('episode_recorder')
        self.config = config
        self.store = store
        self.task = ''
        self.next_name = ''
        self.last_error = ''
        self.processing = set()
        self.session = None
        self.episode = None
        self._lock = threading.Lock()
        self._busy = False
        self._latched = deque(maxlen=LATCHED_CACHE_SIZE)
        self._generic = {}
        self._state_rate = RateMeter()
        self._state_count = 0
        self._teleop = {}
        self._record_button_down = False
        self._last_toggle = 0.0
        self._replay_threads = []

        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=30,
                               reliability=ReliabilityPolicy.RELIABLE)
        latched_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                 reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cameras = {}
        for camera in config.cameras:
            feed = CameraFeed(camera.name, camera.serial)
            self.cameras[camera.name] = feed
            prefix = '/cameras/%s' % camera.name
            group = MutuallyExclusiveCallbackGroup()
            self.create_subscription(Image, prefix + '/color/image_raw',
                                     lambda m, f=feed: self._on_color(f, m), image_qos,
                                     callback_group=group)
            self.create_subscription(CameraInfo, prefix + '/color/camera_info',
                                     lambda m, f=feed: setattr(f, 'color_info', m), 10,
                                     callback_group=group)
            if config.depth:
                depth_group = MutuallyExclusiveCallbackGroup()
                self.create_subscription(Image, prefix + '/aligned_depth_to_color/image_raw',
                                         lambda m, f=feed: self._on_depth(f, m), image_qos,
                                         callback_group=depth_group)
                self.create_subscription(CameraInfo, prefix + '/aligned_depth_to_color/camera_info',
                                         lambda m, f=feed: setattr(f, 'depth_info', m), 10,
                                         callback_group=depth_group)
        control_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Joy, '/teleop/joy', self._on_joy, 100, callback_group=control_group)
        self.create_subscription(String, '/teleop/session', self._on_session, latched_qos,
                                 callback_group=control_group)
        self._raw_group = MutuallyExclusiveCallbackGroup()
        self._state_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0, self._discover_topics, callback_group=control_group)
        self._discover_topics()

    # ---------------------------------------------------------------- inputs

    def _on_color(self, feed, msg):
        now = time.time_ns()
        feed.color_rate.tick(now)
        feed.width, feed.height, feed.encoding = msg.width, msg.height, msg.encoding
        feed.last_color_ns = now
        feed.set_latest(msg)
        episode = self.episode
        if episode is not None:
            episode.write_frame(feed.name, 'color', msg, stamp_ns(msg.header.stamp), now)

    def _on_depth(self, feed, msg):
        now = time.time_ns()
        feed.depth_rate.tick(now)
        episode = self.episode
        if episode is not None:
            episode.write_frame(feed.name, 'depth', msg, stamp_ns(msg.header.stamp), now)

    def _on_joy(self, msg):
        button = self.config.quest_record_button if msg.header.frame_id == 'quest' else self.config.record_button
        pressed = button < len(msg.buttons) and msg.buttons[button] != 0
        if pressed and not self._record_button_down and time.monotonic() - self._last_toggle > 0.5:
            self._last_toggle = time.monotonic()
            threading.Thread(target=self.toggle, daemon=True).start()
        self._record_button_down = pressed

    def _on_session(self, msg):
        try:
            self.session = json.loads(msg.data)
        except ValueError:
            self.session = {'raw': msg.data}

    def _discover_topics(self):
        for topic, types in self.get_topic_names_and_types():
            if topic in self._generic or topic in SKIP_TOPICS or not types:
                continue
            type_name = types[0]
            if type_name in IMAGE_TYPES:
                self._generic[topic] = None
                continue
            publishers = self.get_publishers_info_by_topic(topic)
            if not publishers:
                continue
            reliable = all(p.qos_profile.reliability == ReliabilityPolicy.RELIABLE for p in publishers)
            latched = any(p.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for p in publishers)
            try:
                message_type = get_message(type_name)
            except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                self.get_logger().warning('not recording %s (%s): %s' % (topic, type_name, exc))
                self._generic[topic] = None
                continue
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=2000 if topic == STATE_TOPIC else 100,
                reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE)
            group = self._state_group if topic == STATE_TOPIC else self._raw_group
            subscription = self.create_subscription(
                message_type, topic,
                lambda data, t=topic, n=type_name, l=latched: self._on_raw(t, n, l, data),
                qos, callback_group=group, raw=True)
            self._generic[topic] = (type_name, latched, subscription)

    def _on_raw(self, topic, type_name, latched, data):
        now = time.time_ns()
        if latched:
            self._latched.append((topic, type_name, data, now))
        if topic == STATE_TOPIC:
            self._state_rate.tick(now)
            self._state_count += 1
            if self._state_count % 50 == 1:  # 20 Hz status for the web UI
                self._update_teleop_status(data)
        episode = self.episode
        if episode is not None:
            episode.bag.write(topic, type_name, data, now)

    def _update_teleop_status(self, data):
        state = deserialize_message(data, TeleopState)
        mode = state.robot_mode
        self._teleop = {
            'ee_xyz': [round(float(v), 4) for v in state.o_t_ee[12:15]],
            'gripper_width': round(float(state.gripper_width), 4),
            'gripper_command': ['none', 'open', 'grasp'][min(state.gripper_command, 2)],
            'robot_mode': ROBOT_MODES[mode] if mode < len(ROBOT_MODES) else str(mode),
            'deadman': bool(state.deadman),
            'speed_scale': float(state.speed_scale),
            'dropped': int(state.dropped),
            'errors': state.current_errors,
        }

    # ---------------------------------------------------------------- control

    def toggle(self):
        return self.stop() if self.episode is not None else self.start()

    def start(self):
        with self._lock:
            if self.episode is not None or self._busy:
                return {'ok': False, 'error': 'already recording'}
            free = shutil.disk_usage(self.store.root).free
            if free < MIN_FREE_BYTES:
                self.last_error = 'only %.1f GB free in %s; not recording' % (free / 1e9, self.store.root)
                return {'ok': False, 'error': self.last_error}
            path = None
            try:
                episode_id, path, meta = self.store.create(self.next_name, self.task)
                self.episode = ActiveEpisode(episode_id, path, meta, self.config, list(self._latched))
                if self.episode.bag.storage != self.config.bag_storage:
                    self.get_logger().warning('bag storage %s unavailable, using %s (install '
                                              'ros-humble-rosbag2-storage-mcap for mcap)'
                                              % (self.config.bag_storage, self.episode.bag.storage))
                    self.config.bag_storage = self.episode.bag.storage
            except Exception as exc:
                if path is not None:  # remove the folder this attempt just created
                    shutil.rmtree(path, ignore_errors=True)
                    self.store._changed()
                self.last_error = 'could not start recording: %s' % exc
                self.get_logger().error(self.last_error)
                return {'ok': False, 'error': self.last_error}
            self.next_name = ''
            self.last_error = ''
            self.get_logger().info('recording %s' % episode_id)
            return {'ok': True, 'id': episode_id}

    def stop(self):
        with self._lock:
            episode = self.episode
            if episode is None or self._busy:
                return {'ok': False, 'error': 'not recording'}
            self._busy = True
            self.episode = None  # callbacks stop feeding it from here on
        try:
            end_ns = time.time_ns()
            errors = episode.close()
            self._finalize(episode, end_ns, errors)
        finally:
            with self._lock:
                self._busy = False
        self.processing.add(episode.id)
        thread = threading.Thread(target=self._make_replay, args=(episode,), daemon=True)
        thread.start()
        self._replay_threads = [t for t in self._replay_threads if t.is_alive()] + [thread]
        self.get_logger().info('saved %s' % episode.id)
        return {'ok': True, 'id': episode.id}

    def _finalize(self, episode, end_ns, errors):
        meta = episode.meta
        cameras = {}
        for name, feed in self.cameras.items():
            streams = {}
            for stream in ('color', 'depth'):
                writer = episode.writers.get((name, stream))
                if writer is None:
                    continue
                duration_s = ((writer.last_stamp_ns or 0) - (writer.first_stamp_ns or 0)) / 1e9
                streams[stream] = {
                    'file': '%s_%s.%s' % (name, stream, 'mp4' if stream == 'color' else 'mkv'),
                    'frames_csv': '%s_%s_frames.csv' % (name, stream),
                    'frames': writer.frames,
                    'dropped': writer.dropped,
                    'width': writer.width,
                    'height': writer.height,
                    'ros_encoding': writer.encoding,
                    'first_stamp_ns': writer.first_stamp_ns,
                    'last_stamp_ns': writer.last_stamp_ns,
                    'measured_fps': round((writer.frames - 1) / duration_s, 2) if duration_s > 0 else None,
                }
            cameras[name] = {
                'serial': feed.serial,
                'streams': streams,
                'color_camera_info': message_to_ordereddict(feed.color_info) if feed.color_info else None,
                'depth_camera_info': message_to_ordereddict(feed.depth_info) if feed.depth_info else None,
            }
            if not streams:
                errors.append('no frames from camera %s' % name)
        state_messages = episode.bag.topics.get(STATE_TOPIC, 0)
        if state_messages == 0:
            errors.append('no /teleop/state messages: gamepad_teleop was not running')
        recorder_config = dataclasses.asdict(self.config)
        recorder_config['data_dir'] = str(self.config.data_dir)
        meta.update({
            'status': 'processing',
            'start_ns': episode.start_ns,
            'end_ns': end_ns,
            'duration_s': round((end_ns - episode.start_ns) / 1e9, 3),
            'cameras': cameras,
            'bag': {'path': 'bag', 'storage': episode.bag.storage, 'topics': dict(episode.bag.topics)},
            'teleop_session': self.session,
            'recorder_config': recorder_config,
            'warnings': errors,
        })
        write_json(episode.path / 'meta.json', meta)
        self.store._changed()

    def _make_replay(self, episode):
        warnings = []
        try:
            inputs = [episode.path / ('%s_color.mp4' % name) for name in self.cameras
                      if (name, 'color') in episode.writers and episode.writers[(name, 'color')].frames > 0]
            if inputs:
                make_side_by_side(inputs, episode.path / 'replay.mp4')
        except Exception as exc:
            warnings.append(str(exc))
        finally:
            # Edits are refused while status is 'processing', so the folder is unchanged.
            meta = read_json(episode.path / 'meta.json')
            meta['status'] = 'ready'
            meta['warnings'] = meta.get('warnings', []) + warnings
            write_json(episode.path / 'meta.json', meta)
            self.processing.discard(episode.id)
            self.store._changed()

    def wait_for_processing(self):
        for thread in self._replay_threads:
            thread.join()

    # ---------------------------------------------------------------- status

    def status(self):
        episode = self.episode
        cameras = {}
        now = time.time_ns()
        for name, feed in self.cameras.items():
            frames = None
            if episode is not None:
                writer = episode.writers.get((name, 'color'))
                frames = writer.frames if writer else 0
            cameras[name] = {
                'serial': feed.serial,
                'color_fps': round(feed.color_rate.rate(), 1),
                'depth_fps': round(feed.depth_rate.rate(), 1),
                'width': feed.width,
                'height': feed.height,
                'age_s': round((now - feed.last_color_ns) / 1e9, 2) if feed.last_color_ns else None,
                'recorded_frames': frames,
            }
        state_rate = self._state_rate.rate()
        disk = shutil.disk_usage(self.store.root)
        return {
            'recording': episode is not None,
            'busy': self._busy,
            'episode': {'id': episode.id, 'name': episode.meta['name'],
                        'elapsed_s': round((now - episode.start_ns) / 1e9, 1),
                        'bag_backlog': episode.bag.backlog()} if episode else None,
            'task': self.task,
            'next_name': self.next_name,
            'cameras': cameras,
            'teleop': dict(self._teleop, state_hz=round(state_rate), connected=state_rate > 0),
            'record_button': self.config.record_button,
            'disk_free_gb': round(disk.free / 1e9, 1),
            'data_dir': str(self.store.root),
            'processing': sorted(self.processing),
            'episodes_version': self.store.version,
            'last_error': self.last_error,
        }
