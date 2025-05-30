from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='line_followerxime',
            executable='camera',
            name='camera_subscriber_node',
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(
                    os.getenv('HOME'),
                    'ros2_ws', 'src', 'line_followerxime', 'line_followerxime', 'light_detector.py'
                )
            ],
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(
                    os.getenv('HOME'),
                    'ros2_ws', 'src', 'line_followerxime', 'line_followerxime', 'line_follower.py'
                )
            ],
            output='screen'
        )
    ])
