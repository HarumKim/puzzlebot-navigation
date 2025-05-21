#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class SmartFollower(Node):
    def __init__(self):
        super().__init__('smart_follower')
        self.debug = True

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(Image, '/video_source/raw', self.image_callback, 10)
        self.bridge = CvBridge()

        # Estado del semáforo
        self.green_light_received = False
        self.light_color = "UNKNOWN"

        # PID embebido
        self.Kp = 0.8
        self.Ki = 0.1
        self.Kd = 0.25
        self.max_output = math.radians(40)

        self.setpoint = 0.0
        self.last_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

        # Control de velocidad
        self.max_throttle = 0.2
        self.center_weight = 0.7
        self.angle_weight = 0.2

        self.get_logger().info("🤖 SmartFollower iniciado (PID incluido + detección unificada).")

        if self.debug:
            cv2.namedWindow("Línea - DEBUG", cv2.WINDOW_NORMAL)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.light_color = self.detect_traffic_light(hsv)
        if self.light_color == "GREEN":
            self.green_light_received = True

        throttle, yaw = self.follow_line(frame)

        if not self.green_light_received:
            self.get_logger().info("🛑 Esperando luz VERDE para iniciar...")
            throttle = 0.0
            yaw = 0.0

        # Limitar las velocidades antes de publicar
        yaw = max(-self.max_output, min(self.max_output, yaw))
        throttle = max(0.0, min(self.max_throttle, throttle))

        twist = Twist()
        twist.linear.x = float(throttle)
        twist.angular.z = float(yaw)
        self.cmd_pub.publish(twist)

    def detect_traffic_light(self, hsv_img):
        red1 = cv2.inRange(hsv_img, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv_img, (170, 120, 70), (180, 255, 255))
        red = red1 | red2

        yellow = cv2.inRange(hsv_img, (20, 100, 100), (40, 255, 255))
        green = cv2.inRange(hsv_img, (36, 100, 50), (85, 255, 255))

        if cv2.countNonZero(green) > 300:
            return "GREEN"
        elif cv2.countNonZero(yellow) > 300:
            return "YELLOW"
        elif cv2.countNonZero(red) > 300:
            return "RED"
        else:
            return "UNKNOWN"

    def follow_line(self, frame):
        if frame is None:
            return 0.0, 0.0

        h, w = frame.shape[:2]
        roi_height_start = int(h * 0.6)
        roi_width_margin = 50
        x1 = roi_width_margin
        x2 = w - roi_width_margin
        roi = frame[roi_height_start:, 0:]
        #roi = frame[roi_height_start:, x1:x2]

        frame_center_x = w / 2

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 3000]

        throttle, yaw = 0.0, 0.0
        if contours:
            def line_score(c):
                _, _, angle, cx, cy = self.get_contour_line(c)
                center_score = 1.0 - (abs(cx - frame_center_x) / frame_center_x)
                max_angle = 80
                angle_score = 1.0 - min(abs(angle), max_angle) / max_angle
                total_score = (self.center_weight * center_score) + (self.angle_weight * angle_score)
                return total_score

            scored_contours = [(c, line_score(c)) for c in contours]
            scored_contours.sort(key=lambda x: x[1], reverse=True)

            line_contour = scored_contours[0][0]
            best_score = scored_contours[0][1]

            _, _, angle, cx, cy = self.get_contour_line(line_contour)
            normalized_x = (cx - frame_center_x) / frame_center_x
            yaw = self.compute_pid(normalized_x)

            alignment = 1 - abs(normalized_x)
            align_thres = 0.15
            throttle = self.max_throttle * ((alignment - align_thres) / (1 - align_thres)) if alignment >= align_thres else 0

            self.get_logger().info(f"[THROTTLE] alignment={alignment:.3f}, align_thres=0.3, raw_throttle={throttle:.3f}")
            self.get_logger().info(f"[CMD] throttle_limited={throttle:.3f}, yaw_limited={yaw:.3f}")



            if self.debug:
                self.draw_debug_info(roi, scored_contours, line_contour, best_score, cx, cy)
        else:
            if self.debug:
                cv2.imshow("Línea - DEBUG", roi)
                cv2.waitKey(1)

        return throttle, yaw

    def draw_debug_info(self, roi, scored_contours, best_contour, best_score, cx, cy):
        for c, score in scored_contours[1:]:
            cv2.drawContours(roi, [c], -1, (0, 0, 255), 2)
        cv2.drawContours(roi, [best_contour], -1, (0, 255, 0), 2)
        score_text = f"Selected: Score={best_score:.2f}"
        cv2.putText(roi, score_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.line(roi, (int(cx), 0), (int(cx), roi.shape[0]), (255, 0, 0), 2)
        for c, _ in scored_contours:
            _, _, angle, cx_c, cy_c = self.get_contour_line(c)
            center_score = 1.0 - (abs(cx_c - roi.shape[1] / 2) / (roi.shape[1] / 2))
            angle_score = 1.0 - min(abs(angle), 80) / 80
            total_score = (self.center_weight * center_score) + (self.angle_weight * angle_score)
            debug_text = f"C:{center_score:.2f} A:{angle_score:.2f} T:{total_score:.2f}"
            cv2.putText(roi, debug_text, (int(cx_c), int(cy_c) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Línea - DEBUG", roi)
        cv2.waitKey(1)

    def get_contour_line(self, c, fix_vert=True):
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle = math.degrees(math.atan2(vy, vx))
        if fix_vert:
            angle = angle - 90 * np.sign(angle)
        return None, None, angle, cx, cy

    def compute_pid(self, current_value):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.prev_time if self.prev_time else 0.1
        self.prev_time = now
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return max(-self.max_output, min(self.max_output, output))

def main(args=None):
    rclpy.init(args=args)
    node = SmartFollower()
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
