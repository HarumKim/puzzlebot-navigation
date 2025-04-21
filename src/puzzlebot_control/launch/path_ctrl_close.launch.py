from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='puzzlebot_control',
            executable='odometry',
            name='odometry_node',
            output='screen'
        ),
        Node(
            package='puzzlebot_control',
            executable='close_loop_ctrl',
            name='controller_node',
            output='screen'
        ),
        Node(
            package='puzzlebot_control',
            executable='path_gen_close',
            name='path_node',
            output='screen'
        ),
    ])
