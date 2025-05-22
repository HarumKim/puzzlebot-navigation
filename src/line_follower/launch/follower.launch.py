from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='line_follower',
            executable='camera',
            name='camera_subscriber_node',
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(
                    os.getenv('HOME'),
                    'ros2_ws', 'src', 'line_follower', 'line_follower', 'light_detector.py'
                )
            ],
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(
                    os.getenv('HOME'),
                    'ros2_ws', 'src', 'line_follower', 'line_follower', 'prueba1.py'
                )
            ],
            output='screen'
        )
    ])
