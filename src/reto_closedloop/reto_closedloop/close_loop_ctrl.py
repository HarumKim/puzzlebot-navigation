import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32, Bool
import math
import time

class ClosedLoopPIDController(Node):
    def __init__(self):
        super().__init__('closed_loop_pid_ctrl')

        # PID gains
        self.Kp_lin = 1.0
        self.Ki_lin = 1.5
        self.Kd_lin = 0.05

        self.Kp_ang = 4.0
        self.Ki_ang = 0.1
        self.Kd_ang = 0.01

        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

        self.target = {'x': 2.0, 'y': 2.0, 'theta': 0.0}
        self.new_target_received = False
        self.target_alcanzado = False

        # PID state
        self.prev_error_lin = 0.0
        self.integral_lin = 0.0

        self.prev_error_ang = 0.0
        self.integral_ang = 0.0

        self.prev_time = self.get_clock().now()

        # Subscriptions
        self.create_subscription(Float32, '/odom_x', self.x_callback, 10)
        self.create_subscription(Float32, '/odom_y', self.y_callback, 10)
        self.create_subscription(Float32, '/odom_psi', self.psi_callback, 10)
        self.create_subscription(PoseStamped, '/path_pose', self.pose_callback, 10)

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_reached = self.create_publisher(Bool, '/goal_reached', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)

    def x_callback(self, msg): self.x = msg.data
    def y_callback(self, msg): self.y = msg.data
    def psi_callback(self, msg): self.psi = msg.data

    def pose_callback(self, msg):
        self.target['x'] = msg.pose.position.x
        self.target['y'] = msg.pose.position.y
        z = msg.pose.orientation.z
        w = msg.pose.orientation.w
        self.target['theta'] = math.atan2(2.0 * z * w, 1.0 - 2.0 * z * z)

        self.new_target_received = True
        self.target_alcanzado = False
        self.integral_lin = 0.0
        self.integral_ang = 0.0
        self.prev_error_lin = 0.0
        self.prev_error_ang = 0.0
        self.prev_time = self.get_clock().now()
        self.get_logger().info(f"🎯 New target received: x={self.target['x']:.2f}, y={self.target['y']:.2f}, θ={self.target['theta']:.2f} rad")

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def control_loop(self):
        if not self.new_target_received:
            return

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt == 0:
            return

        dx = self.target['x'] - self.x
        dy = self.target['y'] - self.y
        rho = math.hypot(dx, dy)
        alpha = self.normalize_angle(math.atan2(dy, dx) - self.psi)

        # PID for linear velocity
        error_lin = rho
        self.integral_lin += error_lin * dt
        derivative_lin = (error_lin - self.prev_error_lin) / dt
        v = self.Kp_lin * error_lin + self.Ki_lin * self.integral_lin + self.Kd_lin * derivative_lin

        # PID for angular velocity
        error_ang = alpha
        self.integral_ang += error_ang * dt
        derivative_ang = (error_ang - self.prev_error_ang) / dt
        w = self.Kp_ang * error_ang + self.Ki_ang * self.integral_ang + self.Kd_ang * derivative_ang

        self.prev_error_lin = error_lin
        self.prev_error_ang = error_ang
        self.prev_time = now

        # Limitar velocidades
        v = max(min(v, 0.3), -0.3)
        w = max(min(w, 1.5), -1.5)

        # Publicar mensaje de velocidad
        cmd = Twist()
        # Tolerancias
        pos_tol = 0.03   # posición en metros
        ang_tol = 0.05   # orientación en radianes

        # Control condicional
        if rho >= pos_tol:
            # Ir al punto
            cmd.linear.x = v
            cmd.angular.z = w
        else:
            # Ya en posición, alinear orientación
            theta_error = self.normalize_angle(self.target['theta'] - self.psi)

            error_ang = theta_error
            self.integral_ang += error_ang * dt
            derivative_ang = (error_ang - self.prev_error_ang) / dt
            w_theta = self.Kp_ang * error_ang + self.Ki_ang * self.integral_ang + self.Kd_ang * derivative_ang
            w_theta = max(min(w_theta, 1.5), -1.5)

            if abs(theta_error) < ang_tol:
                # Llegó completamente al objetivo
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                if not self.target_alcanzado:
                    self.get_logger().info('✅ Target reached with orientation.')
                    self.target_alcanzado = True
                    self.pub_reached.publish(Bool(data=True))
            else:
                # Alinear orientación
                cmd.linear.x = 0.0
                cmd.angular.z = w_theta


        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
