#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32, Bool, String
import math
import time

class ClosedLoopPIDController(Node):
    def __init__(self):
        super().__init__('closed_loop_pid_controller')

        # PID gains
        self.Kp_lin = 2.0
        self.Ki_lin = 1.5
        self.Kd_lin = 0.05

        self.Kp_ang = 5.0
        self.Ki_ang = 0.5
        self.Kd_ang = 0.01
        
        self.Kp_cross = 3.0  # Cross-track error correction

        # Posición y orientación actual
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

        # Objetivo
        self.target = {'x': 2.0, 'y': 2.0, 'theta': 0.0}
        self.start_pose = {'x': 0.0, 'y': 0.0}
        self.new_target_received = False
        self.target_reached = False
        
        # Fases
        self.PHASE_ALIGNING = 0
        self.PHASE_MOVING = 1
        self.PHASE_FINAL_ALIGN = 2
        self.current_phase = self.PHASE_ALIGNING
        
        # Tiempo
        self.phase_start_time = None
        self.phase_timeout = 5.0
        self.stuck_count = 0
        self.prev_position = (0.0, 0.0)
        self.movement_threshold = 0.01

        # PID states
        self.prev_error_lin = 0.0
        self.integral_lin = 0.0
        self.prev_error_ang = 0.0
        self.integral_ang = 0.0
        self.prev_time = self.get_clock().now()

        # Tolerancias
        self.pos_tolerance = 0.03
        self.ang_tolerance = 0.15
        self.initial_ang_tolerance = 0.3

        # Semáforo
        self.light_state = "UNKNOWN"
        self.create_subscription(String, '/light', self.light_callback, 10)

        # Subscripciones
        self.create_subscription(Float32, '/odom_x', self.x_callback, 10)
        self.create_subscription(Float32, '/odom_y', self.y_callback, 10)
        self.create_subscription(Float32, '/odom_psi', self.psi_callback, 10)
        self.create_subscription(PoseStamped, '/path_pose', self.pose_callback, 10)

        # Publicadores
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_reached = self.create_publisher(Bool, '/goal_reached', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

    def light_callback(self, msg):
        if msg.data in ["RED", "YELLOW", "GREEN"]:
            self.light_state = msg.data

    def x_callback(self, msg): self.x = msg.data
    def y_callback(self, msg): self.y = msg.data
    def psi_callback(self, msg): self.psi = msg.data

    def pose_callback(self, msg):
        self.target['x'] = msg.pose.position.x
        self.target['y'] = msg.pose.position.y
        z = msg.pose.orientation.z
        w = msg.pose.orientation.w
        self.target['theta'] = math.atan2(2.0 * z * w, 1.0 - 2.0 * z * z)

        self.start_pose = {'x': self.x, 'y': self.y}
        self.new_target_received = True
        self.target_reached = False
        self.current_phase = self.PHASE_ALIGNING
        self.phase_start_time = self.get_clock().now()
        self.stuck_count = 0
        self.reset_pid()

        cmd = Twist()
        self.pub_cmd.publish(cmd)

        self.get_logger().info(f"🌟 New target: x={self.target['x']:.2f}, y={self.target['y']:.2f}, θ={self.target['theta']:.2f}")

    def reset_pid(self):
        self.integral_lin = 0.0
        self.integral_ang = 0.0
        self.prev_error_lin = 0.0
        self.prev_error_ang = 0.0
        self.prev_time = self.get_clock().now()

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def calculate_cross_track_error(self):
        dx_path = self.target['x'] - self.start_pose['x']
        dy_path = self.target['y'] - self.start_pose['y']
        path_length = math.hypot(dx_path, dy_path)
        if path_length < 0.0001:
            return 0.0
        nx = dx_path / path_length
        ny = dy_path / path_length
        dx_robot = self.x - self.start_pose['x']
        dy_robot = self.y - self.start_pose['y']
        projection = dx_robot * nx + dy_robot * ny
        proj_x = self.start_pose['x'] + projection * nx
        proj_y = self.start_pose['y'] + projection * ny
        cross_error = math.hypot(self.x - proj_x, self.y - proj_y)
        cross_product = dx_path * dy_robot - dy_path * dx_robot
        return -cross_error if cross_product > 0 else cross_error

    def advance_phase(self):
        self.reset_pid()
        if self.current_phase == self.PHASE_ALIGNING:
            self.current_phase = self.PHASE_MOVING
        elif self.current_phase == self.PHASE_MOVING:
            self.current_phase = self.PHASE_FINAL_ALIGN
        else:
            self.get_logger().info("✅ Target reached (forced).")
            self.target_reached = True
            self.pub_reached.publish(Bool(data=True))
        self.phase_start_time = self.get_clock().now()

    def control_loop(self):
        if not self.new_target_received:
            return

        if self.light_state == "RED" or self.light_state == "UNKNOWN":
            self.pub_cmd.publish(Twist())
            return

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt == 0:
            return

        dx = self.target['x'] - self.x
        dy = self.target['y'] - self.y
        rho = math.hypot(dx, dy)
        path_angle = math.atan2(self.target['y'] - self.start_pose['y'], self.target['x'] - self.start_pose['x'])

        cmd = Twist()

        if self.current_phase == self.PHASE_ALIGNING:
            heading_error = self.normalize_angle(path_angle - self.psi)
            if abs(heading_error) < self.initial_ang_tolerance:
                self.get_logger().info("🔄 Alineación inicial completada.")
                self.current_phase = self.PHASE_MOVING
                self.phase_start_time = now
            else:
                error_ang = heading_error
                self.integral_ang += error_ang * dt
                derivative_ang = (error_ang - self.prev_error_ang) / dt
                w = self.Kp_ang * error_ang + self.Ki_ang * self.integral_ang + self.Kd_ang * derivative_ang

                # Limitar velocidad angular según semáforo
                if self.light_state == "YELLOW":
                    w = max(min(w, 0.5), -0.5)
                elif self.light_state == "GREEN":
                    w = max(min(w, 1.0), -1.0)

                cmd.angular.z = w
                self.prev_error_ang = error_ang

        elif self.current_phase == self.PHASE_MOVING:
            if rho < self.pos_tolerance:
                self.get_logger().info("📍 Posición alcanzada.")
                self.current_phase = self.PHASE_FINAL_ALIGN
                self.phase_start_time = now
                self.reset_pid()
            else:
                cross_error = self.calculate_cross_track_error()
                error_lin = rho
                self.integral_lin += error_lin * dt
                derivative_lin = (error_lin - self.prev_error_lin) / dt
                v = self.Kp_lin * error_lin + self.Ki_lin * self.integral_lin + self.Kd_lin * derivative_lin

                heading_error = self.normalize_angle(path_angle - self.psi)
                cross_track_correction = math.atan2(self.Kp_cross * cross_error, 0.5)
                combined_error = heading_error + cross_track_correction
                self.integral_ang += combined_error * dt
                derivative_ang = (combined_error - self.prev_error_ang) / dt
                w = self.Kp_ang * combined_error + self.Ki_ang * self.integral_ang + self.Kd_ang * derivative_ang

                # Ajuste de velocidades según semáforo
                if self.light_state == "YELLOW":
                    v = max(min(v, 0.05), -0.05)
                    w = max(min(w, 0.5), -0.5)
                elif self.light_state == "GREEN":
                    v = max(min(v, 0.1), -0.1)
                    w = max(min(w, 1.0), -1.0)

                if abs(combined_error) > math.radians(45):
                    v = 0.0
                elif abs(combined_error) > math.radians(20):
                    v *= 0.3
                if abs(cross_error) > 0.05:
                    v = 0.0

                cmd.linear.x = v
                cmd.angular.z = w
                self.prev_error_lin = error_lin
                self.prev_error_ang = combined_error

        elif self.current_phase == self.PHASE_FINAL_ALIGN:
            theta_error = self.normalize_angle(self.target['theta'] - self.psi)
            if abs(theta_error) < self.ang_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                if not self.target_reached:
                    self.get_logger().info("✅ Objetivo final alcanzado con orientación.")
                    self.target_reached = True
                    self.pub_reached.publish(Bool(data=True))
            else:
                error_ang = theta_error
                self.integral_ang += error_ang * dt
                derivative_ang = (error_ang - self.prev_error_ang) / dt
                w = self.Kp_ang * error_ang + self.Ki_ang * self.integral_ang + self.Kd_ang * derivative_ang

                # Limitar velocidad angular según semáforo
                if self.light_state == "YELLOW":
                    w = max(min(w, 0.5), -0.5)
                elif self.light_state == "GREEN":
                    w = max(min(w, 1.0), -1.0)

                cmd.angular.z = w
                cmd.linear.x = 0.0
                self.prev_error_ang = error_ang

        self.prev_time = now
        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
