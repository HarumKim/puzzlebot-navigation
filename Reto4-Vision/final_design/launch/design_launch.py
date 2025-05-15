#Packages to get address of the YAML file
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    cv_node = Node(name="cv_node",
                          package='final_design',
                          executable='cv',
                          emulate_tty=True,
                          output='screen',
        )
    
    servo_node = Node(name="servo_visualizer_node",
                          package='final_design',
                          executable='servo_visualizer',
                          emulate_tty=True,
                          output='screen',
        )
    
    l_d = LaunchDescription([cv_node, servo_node])

    return l_d