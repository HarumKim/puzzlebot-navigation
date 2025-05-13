from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='reto_closedloop',
            executable='odometry',
            name='odometry_node',
            output='screen'
        ),
        Node(
            package='reto_closedloop',
            executable='close_loop_ctrl',
            name='controller_node',
            output='screen'
        ),
        Node(
            package='reto_closedloop',
            executable='path_gen_close',
            name='path_node',
            output='screen'
        ),
        Node(
            package='reto_closedloop',
            executable='light_detector',
            name='light_detector_node',
            output='screen'
        ),
    ])
