"""Both RealSense cameras + the episode recorder and its web UI.

    ros2 launch franka_teleop recording.launch.py
    ros2 launch franka_teleop recording.launch.py config:=/path/to/recording.yaml

gamepad_teleop runs separately in its own terminal (it needs the deadman and Ctrl-C).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml


def _nodes(context):
    config_path = LaunchConfiguration('config').perform(context)
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    camera = config.get('camera', {})
    profile = '%d,%d,%d' % (camera.get('width', 640), camera.get('height', 480), camera.get('fps', 30))
    depth = bool(camera.get('depth', True))
    imu = bool(camera.get('imu', False))
    initial_reset = bool(camera.get('initial_reset', True))
    nodes = []
    for entry in config['cameras']:
        name = str(entry['name'])
        nodes.append(Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='cameras',
            name=name,
            output='screen',
            parameters=[{
                'serial_no': ParameterValue(str(entry['serial']), value_type=str),
                'camera_name': name,
                'enable_color': True,
                'rgb_camera.color_profile': profile,
                'enable_depth': depth,
                'depth_module.depth_profile': profile,
                'align_depth.enable': depth,
                'enable_sync': True,  # color and depth frames of one camera share a timestamp
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': imu,
                'enable_accel': imu,
                'unite_imu_method': 2 if imu else 0,  # one /imu topic, linear interpolation
                'pointcloud.enable': False,
                'initial_reset': initial_reset,  # hardware reset at start: clears stuck streams
                'publish_tf': True,
            }],
        ))
    nodes.append(Node(
        package='franka_teleop',
        executable='recorder',
        name='episode_recorder',
        output='screen',
        arguments=['--config', config_path],
        additional_env={'PYTHONNOUSERSITE': '1'},
        # On Ctrl-C the recorder finishes the running episode (videos, bag, replay)
        # before exiting; give it time instead of launch's default 5 s.
        sigterm_timeout='120',
        sigkill_timeout='10',
    ))
    return nodes


def generate_launch_description():
    default_config = os.path.join(get_package_share_directory('franka_teleop'), 'config', 'recording.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config,
                              description='Recording config (cameras, data folder, web port)'),
        OpaqueFunction(function=_nodes),
    ])
