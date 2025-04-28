import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32, Bool
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
        
        # Cross-track error correction (for path following)
        self.Kp_cross = 3.0  # Proportional gain for cross-track error

        # Current position and orientation
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

        # Target parameters
        self.target = {'x': 2.0, 'y': 2.0, 'theta': 0.0}
        self.start_pose = {'x': 0.0, 'y': 0.0}  # Store starting position for straight line following
        self.new_target_received = False
        self.target_reached = False
        
        # Approach phases
        self.PHASE_ALIGNING = 0     # Initial orientation
        self.PHASE_MOVING = 1       # Moving to target
        self.PHASE_FINAL_ALIGN = 2  # Final orientation
        self.current_phase = self.PHASE_ALIGNING
        
        # Timeout tracking
        self.phase_start_time = None
        self.phase_timeout = 5.0  # 5 seconds timeout for phases
        self.stuck_count = 0
        self.prev_position = (0.0, 0.0)
        self.movement_threshold = 0.01  # Minimum movement to not be considered stuck

        # PID state
        self.prev_error_lin = 0.0
        self.integral_lin = 0.0
        self.prev_error_ang = 0.0
        self.integral_ang = 0.0
        self.prev_time = self.get_clock().now()

        # Tolerances
        self.pos_tolerance = 0.03   # Position tolerance in meters
        self.ang_tolerance = 0.15   # Orientation tolerance in radians
        self.initial_ang_tolerance = 0.3  # Wider tolerance for initial alignment

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

        # Store starting position for straight line reference
        self.start_pose = {'x': self.x, 'y': self.y}
        
        self.new_target_received = True
        self.target_reached = False
        self.current_phase = self.PHASE_ALIGNING
        self.phase_start_time = self.get_clock().now()
        self.stuck_count = 0

        # Reset PID components
        self.reset_pid()

        # Stop robot when receiving new target
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.pub_cmd.publish(cmd)

        self.get_logger().info(f"🌟 New target received: x={self.target['x']:.2f}, y={self.target['y']:.2f}, θ={self.target['theta']:.2f} rad")

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
        """Calculate cross-track error (distance from the robot to the line connecting start and goal)"""
        # Vector from start to goal
        dx_path = self.target['x'] - self.start_pose['x']
        dy_path = self.target['y'] - self.start_pose['y']
        path_length = math.hypot(dx_path, dy_path)
        
        if path_length < 0.0001:  # Very small path, avoid division by zero
            return 0.0
            
        # Normalize path vector
        nx = dx_path / path_length
        ny = dy_path / path_length
        
        # Vector from start to current position
        dx_robot = self.x - self.start_pose['x']
        dy_robot = self.y - self.start_pose['y']
        
        # Calculate projection length
        projection = dx_robot * nx + dy_robot * ny
        
        # Projected point on the path
        proj_x = self.start_pose['x'] + projection * nx
        proj_y = self.start_pose['y'] + projection * ny
        
        # Cross-track error is the distance from robot to projected point
        cross_error = math.hypot(self.x - proj_x, self.y - proj_y)
        
        # Determine sign of error (left or right of path)
        # Cross product of path vector and robot vector
        cross_product = dx_path * dy_robot - dy_path * dx_robot
        
        if cross_product > 0:
            cross_error = -cross_error  # Robot is to the left of the path
            
        return cross_error

    def check_stuck(self):
        """Check if robot is stuck by comparing current position with previous"""
        current_pos = (self.x, self.y)
        distance_moved = math.hypot(current_pos[0] - self.prev_position[0], current_pos[1] - self.prev_position[1])
        
        # Check if we've barely moved
        now = self.get_clock().now()
        elapsed = (now - self.phase_start_time).nanoseconds / 1e9
        
        if elapsed > 2.0 and distance_moved < self.movement_threshold:
            self.stuck_count += 1
            if self.stuck_count > 10:  # If stuck for 10 consecutive checks
                self.get_logger().warn("Robot appears to be stuck! Switching phases")
                self.advance_phase()
                self.stuck_count = 0
        else:
            self.stuck_count = 0
            
        # Set previous position for next check
        self.prev_position = current_pos
        
        # Check for timeout
        if elapsed > self.phase_timeout:
            self.get_logger().warn(f"Phase timeout after {self.phase_timeout} seconds! Advancing phase")
            self.advance_phase()

    def advance_phase(self):
        """Advance to the next phase of approach"""
        self.reset_pid()  # Reset PID when changing phases
        if self.current_phase == self.PHASE_ALIGNING:
            self.current_phase = self.PHASE_MOVING
        elif self.current_phase == self.PHASE_MOVING:
            self.current_phase = self.PHASE_FINAL_ALIGN
        else:
            # We're in final alignment but stuck - just declare success
            self.get_logger().info("⚠️ Force completing navigation after timeout")
            self.target_reached = True
            self.pub_reached.publish(Bool(data=True))
            
        self.phase_start_time = self.get_clock().now()

    def control_loop(self):
        if not self.new_target_received:
            return

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt == 0:
            return

        # Calculate distances and angles
        dx = self.target['x'] - self.x
        dy = self.target['y'] - self.y
        rho = math.hypot(dx, dy)  # Distance to target
        alpha = self.normalize_angle(math.atan2(dy, dx) - self.psi)  # Angle between robot heading and target

        # Calculate desired path heading (from start to goal)
        path_angle = math.atan2(
            self.target['y'] - self.start_pose['y'], 
            self.target['x'] - self.start_pose['x']
        )

        # Check if we're stuck
        self.check_stuck()

        cmd = Twist()
        
        # State machine for approaching target
        if self.current_phase == self.PHASE_ALIGNING:
            # Initial alignment phase - align with the path direction, not just toward target
            # This helps with straight-line following
            heading_error = self.normalize_angle(path_angle - self.psi)
            
            if abs(heading_error) < self.initial_ang_tolerance:
                self.get_logger().info("✓ Initial alignment complete")
                self.current_phase = self.PHASE_MOVING
                self.phase_start_time = now
            else:
                # PID for angular velocity
                error_ang = heading_error
                self.integral_ang += error_ang * dt
                derivative_ang = (error_ang - self.prev_error_ang) / dt
                w = self.Kp_ang * error_ang + self.Ki_ang * self.integral_ang + self.Kd_ang * derivative_ang
                
                # Limit angular velocity
                w = max(min(w, 1.5), -1.5)
                
                cmd.linear.x = 0.0
                cmd.angular.z = w
                self.prev_error_ang = error_ang
                
        elif self.current_phase == self.PHASE_MOVING:
            # Moving to target phase with path following
            if rho < self.pos_tolerance:
                self.get_logger().info("✓ Position reached")
                self.current_phase = self.PHASE_FINAL_ALIGN
                self.phase_start_time = now
                self.reset_pid()
            else:
                # Calculate cross-track error for straight line following
                cross_error = self.calculate_cross_track_error()
                
                # PID for linear velocity (based on distance to target)
                error_lin = rho
                self.integral_lin += error_lin * dt
                derivative_lin = (error_lin - self.prev_error_lin) / dt
                v = self.Kp_lin * error_lin + self.Ki_lin * self.integral_lin + self.Kd_lin * derivative_lin
                
                # Combine path-following with target-seeking
                # We want to maintain the path heading while correcting for cross-track error
                heading_error = self.normalize_angle(path_angle - self.psi)
                cross_track_correction = math.atan2(self.Kp_cross * cross_error, 0.5)  # 0.5 is a lookahead distance
                
                # Combined angular error
                combined_error = heading_error + cross_track_correction
                
                # PID for angular velocity 
                self.integral_ang += combined_error * dt
                derivative_ang = (combined_error - self.prev_error_ang) / dt
                w = self.Kp_ang * combined_error + self.Ki_ang * self.integral_ang + self.Kd_ang * derivative_ang
                
                # Limit velocities
                v = max(min(v, 0.3), -0.3)
                w = max(min(w, 1.5), -1.5)
                
                # Reduce linear velocity when angle error is large
                if abs(combined_error) > math.radians(45):
                    v = 0.0
                elif abs(combined_error) > math.radians(20):
                    v *= 0.3
                
                # Log significant cross-track errors for debugging
                if abs(cross_error) > 0.05:
                    v = 0.0
                    self.get_logger().debug(f"Cross-track error: {cross_error:.3f}m, Correction: {math.degrees(cross_track_correction):.1f}°")
                
                cmd.linear.x = v
                cmd.angular.z = w
                self.prev_error_lin = error_lin
                self.prev_error_ang = combined_error
                
        elif self.current_phase == self.PHASE_FINAL_ALIGN:
            # Final alignment phase
            theta_error = self.normalize_angle(self.target['theta'] - self.psi)
            
            if abs(theta_error) < self.ang_tolerance:
                # Target completely reached
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                if not self.target_reached:
                    self.get_logger().info('✅ Target reached with orientation.')
                    self.target_reached = True
                    self.pub_reached.publish(Bool(data=True))
            else:
                # Align to final orientation
                error_ang = theta_error
                self.integral_ang += error_ang * dt
                derivative_ang = (error_ang - self.prev_error_ang) / dt
                w = self.Kp_ang * error_ang + self.Ki_ang * self.integral_ang + self.Kd_ang * derivative_ang
                w = max(min(w, 1.5), -1.5)
                
                cmd.linear.x = 0.0
                cmd.angular.z = w
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