#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from custom_interfaces.srv import SetProcessBool

class SmartFollower(Node):
    def __init__(self):
        super().__init__('smart_follower')
        self.debug = True

        # Servicio para habilitar/deshabilitar el seguimiento
        self.simulation_running = False
        self.srv = self.create_service(SetProcessBool, 'EnableProcess', self.set_process_callback)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(Image, '/video_source/raw', self.image_callback, qos_profile)
        self.bridge = CvBridge()

        # Estado del semáforo
        self.light_sub = self.create_subscription(String, '/light', self.light_callback, 10)
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

        self.last_frame_time = time.time()
        self.frame_timeout = 1.0
        self.frame_check_timer = self.create_timer(0.1, self.check_frame_timeout)

        self.get_logger().info("🤖 SmartFollower iniciado (PID incluido + detección unificada).")

        if self.debug:
            cv2.namedWindow("Línea - DEBUG", cv2.WINDOW_NORMAL)
    def check_frame_timeout(self):
        if not self.simulation_running:
            return
        now = time.time()
        if (now - self.last_frame_time) > self.frame_timeout:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.get_logger().warn("⚠️ No frames received recently. Robot stopped for safety.")
            
    def set_process_callback(self, request, response):
        self.simulation_running = request.enable
        if request.enable:
            self.get_logger().info("🟢 Proceso de seguimiento activado")
            response.message = "Simulation Started Successfully"
        else:
            self.get_logger().info("🔴 Proceso de seguimiento detenido")
            response.message = "Simulation Stopped Successfully"
        response.success = True
        return response

    def image_callback(self, msg):
        self.last_frame_time = time.time()
        if not self.simulation_running:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        throttle, yaw = self.follow_line(frame)

        if not self.green_light_received:
            self.get_logger().info("🚩 Esperando luz VERDE para iniciar...")
            throttle = 0.0
            yaw = 0.0

        yaw = max(-self.max_output, min(self.max_output, yaw))
        throttle = max(0.0, min(self.max_throttle, throttle))

        twist = Twist()
        twist.linear.x = float(throttle)
        twist.angular.z = float(yaw)
        self.cmd_pub.publish(twist)

    def light_callback(self, msg):
        if msg.data in ["RED", "YELLOW", "GREEN"]:
            self.light_color = msg.data
            if self.light_color == "GREEN":
                self.green_light_received = True

    def follow_line(self, frame):
        if frame is None:
            return 0.0, 0.0

        h, w = frame.shape[:2]
        roi_height_start = int(h * 0.5)

        # 🔧 Reducción del campo de visión horizontal: recorte desde el centro
        roi_width = int(w * 0.6)  # por ejemplo, usar solo el 50% del ancho
        x_start = int((w - roi_width) / 2)
        x_end = x_start + roi_width

        roi = frame[roi_height_start:, x_start:x_end]
        frame_center_x = roi.shape[1] / 2  # centro de la nueva ROI


        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 3000]

        throttle, yaw = 0.0, 0.0
        if len(contours) >= 3:
            # Ordenar por centro en X
            def get_cx(c):
                M = cv2.moments(c)
                return int(M['m10'] / M['m00']) if M['m00'] != 0 else 0

            contours.sort(key=get_cx)
            line_contour = contours[1]  # tomar la línea del centro
        elif contours:
            line_contour = max(contours, key=cv2.contourArea)
        else:
            if self.debug:
                cv2.imshow("Línea - DEBUG", roi)
                cv2.waitKey(1)
            return 0.0, 0.0

        _, _, angle, cx, cy = self.get_contour_line(line_contour)
        normalized_x = (cx - frame_center_x) / frame_center_x

        # Corrección de sesgo si la cámara está desalineada
        bias = 0.0  # ajusta a -0.03 o 0.03 si lo necesitas
        normalized_x += bias

        yaw = self.compute_pid(normalized_x)

        # Banda muerta para evitar temblores
        if abs(normalized_x) < 0.02:
            yaw = 0.0

        # Límite desigual si quieres restringir el giro más hacia un lado
        if normalized_x > 0:
            yaw = min(yaw, math.radians(25))
        else:
            yaw = max(yaw, -math.radians(35))

        alignment = 1 - abs(normalized_x)
        align_thres = 0.15
        throttle = self.max_throttle * ((alignment - align_thres) / (1 - align_thres)) if alignment >= align_thres else 0

        if self.light_color == "RED":
            throttle = 0.0
            yaw = 0.0
        elif self.light_color == "YELLOW":
            throttle = max(min(throttle, 0.05), -0.05)
            yaw = max(min(yaw, 0.5), -0.5)
        elif self.light_color == "GREEN":
            throttle = max(min(throttle, 0.1), -0.1)
            yaw = max(min(yaw, 1.0), -1.0)
        else:
            throttle = 0.0
            yaw = 0.0

        if self.debug:
            self.draw_debug_info(roi, contours, line_contour, cx, cy)

        return throttle, yaw

    def draw_debug_info(self, roi, contours, best_contour, cx, cy):
        for c in contours:
            cv2.drawContours(roi, [c], -1, (0, 0, 255), 2)
        cv2.drawContours(roi, [best_contour], -1, (0, 255, 0), 2)
        cv2.line(roi, (int(cx), 0), (int(cx), roi.shape[0]), (255, 0, 0), 2)
        cv2.imshow("Línea - DEBUG", roi)
        cv2.waitKey(1)

    def get_contour_line(self, c, fix_vert=True):
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle = math.degrees(math.atan2(vy, vx))
        if fix_vert:
            angle -= 90 * np.sign(angle)
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