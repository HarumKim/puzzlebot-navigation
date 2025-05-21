from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Nodo de la cámara en C++ que publica en /video_source/raw
        Node(
            package='line_follower',
            executable='camera',
            name='camera_subscriber_node',
            output='screen'
        ),

        # Nodo de seguimiento de línea en Python
        Node(
            package='line_follower',
            executable='prueba1',
            name='smart_follower_node',
            output='screen'
        ),
    ])
