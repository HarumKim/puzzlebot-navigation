import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import yaml
import math
import os
from ament_index_python.packages import get_package_share_directory

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def load_waypoints(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data['waypoints']

class PathGenerator(Node):
    def __init__(self):
        super().__init__('path_generator')

        self.publisher = self.create_publisher(PoseStamped, '/path_pose', 10)
        self.create_subscription(Bool, '/goal_reached', self.goal_reached_callback, 10)

        self.current_index = 0
        self.goal_reached = False

        # Cargar waypoints
        pkg_path = get_package_share_directory('reto_closedloop')
        path = os.path.join(pkg_path, 'config', 'path_params.yaml')
        self.waypoints = load_waypoints(path)

        # Calcular theta si no está definido
        for i in range(len(self.waypoints) - 1):
            if 'theta' not in self.waypoints[i] or self.waypoints[i]['theta'] is None:
                dx = self.waypoints[i+1]['x'] - self.waypoints[i]['x']
                dy = self.waypoints[i+1]['y'] - self.waypoints[i]['y']
                self.waypoints[i]['theta'] = math.atan2(dy, dx)
        if 'theta' not in self.waypoints[-1] or self.waypoints[-1]['theta'] is None:
            self.waypoints[-1]['theta'] = self.waypoints[-2]['theta']

        self.get_logger().info(f"🟢 Waypoints cargados: {len(self.waypoints)}")
        self.publish_next_goal()

    def goal_reached_callback(self, msg):
        if msg.data and self.current_index < len(self.waypoints):
            self.get_logger().info(f"✅ Waypoint {self.current_index - 1} alcanzado.")
            self.publish_next_goal()

    def publish_next_goal(self):
        if self.current_index >= len(self.waypoints):
            self.get_logger().info("🥳 Todos los waypoints han sido publicados.")
            return

        wp = self.waypoints[self.current_index]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.pose.position.x = float(wp['x'])
        msg.pose.position.y = float(wp['y'])
        msg.pose.position.z = 0.0

        yaw = float(wp['theta'])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.publisher.publish(msg)
        self.get_logger().info(f"📤 Publicado waypoint {self.current_index}: x={wp['x']:.2f}, y={wp['y']:.2f}")
        self.current_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = PathGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
