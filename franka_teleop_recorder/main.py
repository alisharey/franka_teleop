"""Entry point: ROS node on background threads, web server in the main thread."""

import argparse
import signal
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_share_directory

from .config import load_config
from .episodes import EpisodeStore
from .node import Recorder
from .web import make_server


def main():
    share = get_package_share_directory('franka_teleop')
    parser = argparse.ArgumentParser(description='Episode recorder with web UI')
    parser.add_argument('--config', default=share + '/config/recording.yaml')
    parser.add_argument('--web-dir', default=share + '/web')
    args, _ = parser.parse_known_args()  # ignore the --ros-args ros2 launch appends

    config = load_config(args.config)
    store = EpisodeStore(config.data_dir)
    for episode_id in store.recover_interrupted():
        print('marked interrupted episode %s' % episode_id)

    # Ctrl-C is handled here (KeyboardInterrupt), so a running episode is saved.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # Explicitly: a process started in the background by a script inherits
    # SIGINT as ignored, and Python then installs no handler of its own.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))
    recorder = Recorder(config, store)
    try:
        server = make_server(recorder, store, args.web_dir, config.http_host, config.http_port)
    except OSError as exc:
        recorder.get_logger().error('cannot serve the web UI on %s:%d (%s). Is another recorder '
                                    'running? Otherwise change http_port in the config.'
                                    % (config.http_host, config.http_port, exc.strerror))
        recorder.destroy_node()
        rclpy.try_shutdown()
        raise SystemExit(1)
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(recorder)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    host = 'localhost' if config.http_host in ('127.0.0.1', '0.0.0.0') else config.http_host
    recorder.get_logger().info('web UI: http://%s:%d  (episodes in %s)' % (host, config.http_port, config.data_dir))
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        # Finish saving even if Ctrl-C is pressed again or launch sends SIGTERM.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if recorder.episode is not None:
            recorder.get_logger().info('saving the running episode before exit...')
            recorder.stop()
        recorder.wait_for_processing()
        server.server_close()
        executor.shutdown(timeout_sec=1.0)
        recorder.destroy_node()
        rclpy.try_shutdown()
