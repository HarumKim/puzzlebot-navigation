#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from custom_interfaces.srv import SetProcessBool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.debug = True
        self.simulation_running = False

        self.srv = self.create_service(SetProcessBool, 'EnableProcess', self.set_process_callback)

        self.max_linear_speed = 0.08
        self.max_angular_speed = 1.4

        self.Kp_ang = 0.8
        self.Ki_ang = 0.0
        self.Kd_ang = 0.0
        self.setpoint_ang = 0.0
        self.last_error_ang = 0.0
        self.integral_ang = 0.0
        self.prev_time_ang = time.time()

        self.Kp_lin = 0.8
        self.Ki_lin = 0.13
        self.Kd_lin = 0.0
        self.setpoint_lin = 1.0
        self.last_error_lin = 0.0
        self.integral_lin = 0.0
        self.prev_time_lin = time.time()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.error_pub = self.create_publisher(Float32, '/normalized_error', 10)
        self.line_status_pub = self.create_publisher(String, '/line_detected', 10)
        self.line_view_pub = self.create_publisher(Image, '/line_view', 10)

        self.line_was_detected = None
        self.bridge = CvBridge()
        self.last_normalized_error = 0.0
        self.error_change_threshold = 0.5

        # ✅ Suscripción con QoS BEST_EFFORT
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/video_source/raw/compressed',
            self.compressed_callback,
            qos_profile
        )

        self.get_logger().info("🟢 Line follower (compressed) started")

    def set_process_callback(self, request, response):
        self.simulation_running = request.enable
        response.success = True
        response.message = "Seguimiento activado" if request.enable else "Seguimiento detenido"
        self.get_logger().info(response.message)
        if request.enable:
            self.reset_both_pids()
        return response

    def compressed_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            bird_view = self.bird_eye_view(frame)
            throttle, yaw = self.follow_line(bird_view)

            # Limita velocidades
            yaw = max(-self.max_angular_speed, min(self.max_angular_speed, yaw))
            throttle = max(0.0, min(self.max_linear_speed, throttle))

            # ✅ Solo publica /cmd_vel si está activado
            if self.simulation_running and (throttle != 0.0 or yaw != 0.0):
                twist = Twist()
                twist.linear.x = float(throttle)
                twist.angular.z = float(yaw)
                self.cmd_pub.publish(twist)

            # Visualización y publicación de vista debug
            if self.debug:
                cv2.imshow("Bird View", bird_view)
                cv2.waitKey(1)

            line_img_msg = self.bridge.cv2_to_imgmsg(bird_view, encoding="bgr8")
            line_img_msg.header.stamp = self.get_clock().now().to_msg()
            line_img_msg.header.frame_id = "line_debug_view"
            self.line_view_pub.publish(line_img_msg)

        except Exception as e:
            self.get_logger().error(f"❌ Error en compressed_callback: {e}")

    def bird_eye_view(self, frame):
        h, w = frame.shape[:2]
        src_pts = np.float32([
            [w * 0.25, h * 0.5],
            [w * 0.75, h * 0.5],
            [w * 0.1, h],
            [w * 0.9, h]
        ])
        dst_pts = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(frame, matrix, (w, h))

    def follow_line(self, frame):
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.35):, int(w * 0.05):int(w * 0.95)]
        frame_center_x = roi.shape[1] / 2

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 3000]

        if not contours:
            self.line_status_pub.publish(String(data="not_detected"))
            if self.line_was_detected != False:
                self.get_logger().info("⚠️ Línea no detectada")
                self.line_was_detected = False
            return 0.0, 0.0
        else:
            self.line_status_pub.publish(String(data="detected"))
            if self.line_was_detected != True:
                self.get_logger().info("✅ Línea detectada")
                self.line_was_detected = True

        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                centroids.append(cx)

        avg_cx = sum(centroids) / len(centroids) if centroids else frame_center_x
        normalized_x = (avg_cx - frame_center_x) / frame_center_x

        error_change = abs(normalized_x - self.last_normalized_error)
        if error_change > self.error_change_threshold:
            self.reset_both_pids()
        self.last_normalized_error = normalized_x

        self.error_pub.publish(Float32(data=float(normalized_x)))

        yaw = self.compute_angular_pid(normalized_x)
        alignment = 1 - abs(normalized_x)
        throttle = self.compute_linear_pid(alignment)

        if abs(normalized_x) < 0.02:
            yaw = 0.0

        if self.debug:
            self.draw_debug_info(roi, contours, avg_cx)

        return throttle, yaw

    def compute_angular_pid(self, current_error):
        now = time.time()
        dt = now - self.prev_time_ang if self.prev_time_ang else 0.1
        self.prev_time_ang = now

        error = self.setpoint_ang - current_error
        proportional = self.Kp_ang * error
        self.integral_ang += error * dt
        self.integral_ang = max(-1.0, min(1.0, self.integral_ang))
        integral = self.Ki_ang * self.integral_ang
        derivative = self.Kd_ang * (error - self.last_error_ang) / dt
        self.last_error_ang = error

        return proportional + integral + derivative

    def compute_linear_pid(self, current_alignment):
        now = time.time()
        dt = now - self.prev_time_lin if self.prev_time_lin else 0.1
        self.prev_time_lin = now

        error = self.setpoint_lin - current_alignment
        proportional = self.Kp_lin * error
        self.integral_lin += error * dt
        self.integral_lin = max(-0.5, min(0.5, self.integral_lin))
        integral = self.Ki_lin * self.integral_lin
        derivative = self.Kd_lin * (error - self.last_error_lin) / dt
        self.last_error_lin = error

        base_speed = self.max_linear_speed * 0.8
        throttle = base_speed * (1.0 - abs(error)) + (proportional + integral + derivative) * 0.1
        return max(0.0, min(self.max_linear_speed, throttle))

    def draw_debug_info(self, roi, contours, avg_cx):
        for c in contours:
            cv2.drawContours(roi, [c], -1, (0, 0, 255), 2)
        cv2.line(roi, (int(avg_cx), 0), (int(avg_cx), roi.shape[0]), (255, 0, 0), 2)

    def reset_both_pids(self):
        self.integral_ang = 0.0
        self.last_error_ang = 0.0
        self.prev_time_ang = time.time()
        self.integral_lin = 0.0
        self.last_error_lin = 0.0
        self.prev_time_lin = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.debug:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
