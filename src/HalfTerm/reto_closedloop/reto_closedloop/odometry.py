import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import math

from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class Odometry(Node):
    def __init__(self):
        super().__init__('odometry')

        # Parámetros físicos del robot
        self.r = 0.05    # radio de la rueda en metros
        self.L = 0.19    # distancia entre ruedas en metros
        self.dt = 0.1    # intervalo de tiempo en segundos

        # Estados internos
        self.omega_l = 0.0
        self.omega_r = 0.0
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

        # QoS compatible con micro-ROS
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Subscripciones a velocidades angulares de ruedas
        self.create_subscription(Float32, '/VelocityEncL', self.left_callback, qos_profile)
        self.create_subscription(Float32, '/VelocityEncR', self.right_callback, qos_profile)

        # Publicadores de odometría
        self.pub_x = self.create_publisher(Float32, '/odom_x', 10)
        self.pub_y = self.create_publisher(Float32, '/odom_y', 10)
        self.pub_psi = self.create_publisher(Float32, '/odom_psi', 10)

        # Timer para actualización periódica
        self.create_timer(self.dt, self.update_odometry)
        self.get_logger().info("📍 Nodo 'odometry' iniciado y corriendo con velocidades de encoders (BEST_EFFORT).")

    def left_callback(self, msg):
        self.omega_l = msg.data

    def right_callback(self, msg):
        self.omega_r = msg.data

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def update_odometry(self):
        # Cinemática diferencial
        v = self.r * (self.omega_r + self.omega_l) / 2.0
        w = self.r * (self.omega_r - self.omega_l) / self.L

        # Actualizar pose
        self.x += v * self.dt * math.cos(self.psi)
        self.y += v * self.dt * math.sin(self.psi)
        self.psi += w * self.dt
        self.psi = self.normalize_angle(self.psi)

        # Publicar odometría
        self.pub_x.publish(Float32(data=self.x))
        self.pub_y.publish(Float32(data=self.y))
        self.pub_psi.publish(Float32(data=self.psi))


def main(args=None):
    rclpy.init(args=args)
    node = Odometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
