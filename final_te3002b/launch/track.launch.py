from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='final_te3002b',
            executable='camera',
            name='camera_subscriber_node',
            output='screen'
        ),
         ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(
                    os.getenv('HOME'),
                    'ros2_ws', 'src', 'final_te3002b', 'final_te3002b', 'line_follower.py'
                )
            ],
            output='screen'
        ),

        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(
                    os.getenv('HOME'),
                    'ros2_ws', 'src', 'final_te3002b', 'final_te3002b', 'traffic_light.py'
                )
            ],
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(
                    os.getenv('HOME'),
                    'ros2_ws', 'src', 'final_te3002b', 'final_te3002b', 'traffic_signs.py'
                )
            ],
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(
                    os.getenv('HOME'),
                    'ros2_ws', 'src', 'final_te3002b', 'final_te3002b', 'state_machine.py'
                )
            ],
            output='screen'
        )
    ])