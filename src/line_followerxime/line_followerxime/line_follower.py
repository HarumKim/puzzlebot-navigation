#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
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

        self.max_linear_speed = 0.08
        self.max_angular_speed = 1.4

        self.get_logger().info(f"📈 Límites de velocidad - Linear: {self.max_linear_speed:.2f} m/s | Angular: {self.max_angular_speed:.2f} rad/s")

        # Servicio para habilitar/deshabilitar el seguimiento
        self.simulation_running = False
        self.srv = self.create_service(SetProcessBool, 'EnableProcess', self.set_process_callback)
        self.error_pub = self.create_publisher(Float32, '/normalized_error', 10)

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
        self.prev_light_color = "UNKNOWN"
        self.light_color = "UNKNOWN"
        self.waiting_for_green = True
        self.green_released_once = False
        self.waiting_printed = False  # Para evitar spam de consola
        self.last_light_msg = None    # Para detectar repeticiones idénticas

        # Conexión con el modelo YOLO de detección de señales de tránsito
        self.yolo_signal = "N/A"
        self.override_behavior = None
        self.override_timer = None
        self.override_active = False
        self.signal_sub = self.create_subscription(String, '/yolo_signal', self.signal_callback, 10)

        # PID separados para control lineal y angular
        # PID Angular (para seguimiento de línea)
        self.Kp_ang = 0.8
        self.Ki_ang = 0.0
        self.Kd_ang = 0.0
        self.setpoint_ang = 0.0
        self.last_error_ang = 0.0
        self.integral_ang = 0.0
        self.prev_time_ang = time.time()

        # PID Lineal (para control de velocidad basado en alineación)
        self.Kp_lin = 0.8
        self.Ki_lin = 0.13
        self.Kd_lin = 0.0
        self.setpoint_lin = 1.0  # Velocidad objetivo cuando está bien alineado
        self.last_error_lin = 0.0
        self.integral_lin = 0.0
        self.prev_time_lin = time.time()

        self.last_frame_time = time.time()
        self.frame_timeout = 1.0
        self.frame_check_timer = self.create_timer(0.1, self.check_frame_timeout)

        # Variables para detectar cambios bruscos y resetear PID
        self.last_normalized_error = 0.0
        self.error_change_threshold = 0.5  # Si el error cambia más de esto, resetear PID

        if self.debug:
            #cv2.namedWindow("Línea - DEBUG", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Bird's Eye View", cv2.WINDOW_NORMAL)

        self.get_logger().info("🤖 SmartFollower iniciado con PIDs separados.")

    def reset_angular_pid(self):
        """Resetea solo el PID angular"""
        self.integral_ang = 0.0
        self.last_error_ang = 0.0
        self.prev_time_ang = time.time()
        
    def reset_linear_pid(self):
        """Resetea solo el PID lineal"""
        self.integral_lin = 0.0
        self.last_error_lin = 0.0
        self.prev_time_lin = time.time()

    def reset_both_pids(self):
        """Resetea ambos PIDs"""
        self.reset_angular_pid()
        self.reset_linear_pid()
        self.get_logger().info("🔄 PIDs reseteados por cambio brusco en error")

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
        response.success = True
        response.message = "Simulation Started Successfully" if request.enable else "Simulation Stopped Successfully"
        self.get_logger().info("🟢 Proceso de seguimiento activado" if request.enable else "🔴 Proceso de seguimiento detenido")
        
        # Resetear PIDs al cambiar estado
        if request.enable:
            self.reset_both_pids()
            
        return response

    def image_callback(self, msg):
        self.last_frame_time = time.time()
        if not self.simulation_running:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        bird_view = self.bird_eye_view(frame)
        throttle, yaw = self.follow_line(bird_view)

        if not self.green_light_received:
            self.get_logger().info("🚩 Esperando luz VERDE para iniciar...")
            throttle = 0.0
            yaw = 0.0

        yaw = max(-self.max_angular_speed, min(self.max_angular_speed, yaw))
        throttle = max(0.0, min(self.max_linear_speed, throttle))

        twist = Twist()
        twist.linear.x = float(throttle)
        twist.angular.z = float(yaw)
        self.cmd_pub.publish(twist)
    
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
        warped = cv2.warpPerspective(frame, matrix, (w, h))
        return warped
    
    def light_callback(self, msg):
        if msg.data in ["RED", "YELLOW", "GREEN"]:
            # Reiniciar siempre al recibir nueva lectura del semáforo (aunque no cambie)
            self.get_logger().info(f"🟢 Nueva lectura de semáforo: {msg.data} (antes: {self.light_color})")
            
            # Resetear estado de señales siempre
            if self.override_active:
                self.get_logger().info("🔁 Reiniciando lógica de señales por nueva lectura de semáforo.")
            self.override_active = False
            self.override_behavior = None
            self.override_timer = None

            # Actualizar el color
            self.prev_light_color = self.light_color
            self.light_color = msg.data

            if self.light_color == "GREEN":
                self.green_light_received = True

    def signal_callback(self, msg):
        if not self.green_light_received:
            return  # No proceses señales si no hay luz verde

        self.yolo_signal = msg.data
        #self.get_logger().info(f"📸 Señal detectada por YOLO: {self.yolo_signal}")

        if msg.data == "giveWay":
            self.override_behavior = "give_way"
            self.override_active = True
            self.override_timer = time.time()  # Marca el inicio
        elif msg.data == "stop":
            self.override_behavior = "stop"
            self.override_active = True
        elif msg.data == "roadwork":
            self.override_behavior = "slow"
            self.override_active = True
        elif msg.data == "roundabout":
            self.override_behavior = "roundabout"
            self.override_active = True
        elif msg.data == "turnLeft":
            self.override_behavior = "left"
            self.override_active = True
        elif msg.data == "turnRight":
            self.override_behavior = "right"
            self.override_active = True
        elif msg.data == "aheadOnly":
            self.override_behavior = "ahead"
            self.override_active = True
        else:
            self.override_behavior = None
            self.override_active = False

    def follow_line(self, frame):
        if frame is None:
            return 0.0, 0.0

        h, w = frame.shape[:2]
        roi = frame[int(h * 0.35):, int(w * 0.05):int(w * 0.95)]
        frame_center_x = roi.shape[1] / 2

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 3000]

        if not contours:
            if self.debug:
                cv2.imshow("Bird's Eye View", roi)
                cv2.waitKey(1)

            return 0.0, 0.0

        # Obtener centroides de los contornos detectados
        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                centroids.append(cx)

        # Calcular centro promedio
        if centroids:
            avg_cx = sum(centroids) / len(centroids)
        else:
            avg_cx = frame_center_x

        normalized_x = (avg_cx - frame_center_x) / frame_center_x
        
        # Detectar cambios bruscos en el error y resetear PID si es necesario
        error_change = abs(normalized_x - self.last_normalized_error)
        if error_change > self.error_change_threshold:
            self.reset_both_pids()
            
        self.last_normalized_error = normalized_x
        
        # Publicar el error normalizado
        self.error_pub.publish(Float32(data=float(normalized_x)))
        
        # Calcular control angular con PID separado
        yaw = self.compute_angular_pid(normalized_x)
        
        # Calcular alineación para control de velocidad
        alignment = 1 - abs(normalized_x)
        
        # Calcular control lineal con PID separado
        throttle = self.compute_linear_pid(alignment)

        # Aplicar zona muerta para el control angular
        if abs(normalized_x) < 0.02:
            yaw = 0.0
            
        # Limitar velocidades
        yaw = min(max(yaw, -self.max_angular_speed), self.max_angular_speed)

        # 🛑 SEMÁFORO — Prioridad absoluta
        if self.light_color == "RED":
            #self.get_logger().info("🔴 Semáforo en ROJO: Deteniendo.")
            return 0.0, 0.0

        # 🟡 AMARILLO: permitir señales, pero velocidad limitada
        if self.light_color == "YELLOW":
            #self.get_logger().info("🟡  Semáforo en AMARILLO: Desacelerando.")
            throttle = min(throttle, 0.02)

        # 🟢 VERDE: permitir comportamiento normal (ya se aplicó arriba)

        # 🚦 Lógica de señales (solo si GREEN o YELLOW)
        if self.light_color in ["GREEN", "YELLOW"] and self.override_active:
            now = time.time()

            if self.override_behavior == "give_way":
                throttle = 0.0
                yaw = 0.0
                if now - self.override_timer >= 3.0:
                    self.get_logger().info("⏱️ Pausa giveWay finalizada.")
                    self.override_active = False
                    self.override_behavior = None
                else:
                    self.get_logger().info("🚧 Pausa por giveWay en curso.")
                    return throttle, yaw

            elif self.override_behavior == "stop":
                throttle = 0.0
                yaw = 0.0
                self.get_logger().info("🛑 Señal STOP. Detenido.")
            elif self.override_behavior == "slow":
                throttle *= 0.5
                self.get_logger().info("🚧 Roadwork Ahead: reducción de velocidad.")
            elif self.override_behavior == "roundabout":
                throttle *= 0.4
                self.get_logger().info("🔁 Roundabout: velocidad reducida.")
            elif self.override_behavior == "left":
                yaw += 0.5
                self.get_logger().info("↩️ Turn Left Ahead.")
            elif self.override_behavior == "right":
                yaw -= 0.5
                self.get_logger().info("↪️ Turn Right Ahead.")
            elif self.override_behavior == "ahead":
                yaw = 0.0
                self.get_logger().info("⬆️ Ahead Only.")

        if self.debug:
            self.draw_debug_info(roi, contours, avg_cx)

        return throttle, yaw


    def draw_debug_info(self, roi, contours, avg_cx):
        for c in contours:
            cv2.drawContours(roi, [c], -1, (0, 0, 255), 2)
        cv2.line(roi, (int(avg_cx), 0), (int(avg_cx), roi.shape[0]), (255, 0, 0), 2)
        cv2.imshow("Bird's Eye View", roi)
        cv2.waitKey(1)

    def compute_angular_pid(self, current_error):
        """Calcula el control PID para velocidad angular"""
        now = time.time()
        dt = now - self.prev_time_ang
        if dt <= 0:
            dt = 0.1
        
        self.prev_time_ang = now
        error = self.setpoint_ang - current_error  # Error angular
        
        # Término proporcional
        proportional = self.Kp_ang * error
        
        # Término integral
        self.integral_ang += error * dt
        # Anti-windup: limitar integral
        max_integral = 1.0
        self.integral_ang = max(-max_integral, min(max_integral, self.integral_ang))
        integral = self.Ki_ang * self.integral_ang
        
        # Término derivativo
        derivative = self.Kd_ang * (error - self.last_error_ang) / dt
        self.last_error_ang = error
        
        # Salida PID
        output = proportional + integral + derivative
        
        return output

    def compute_linear_pid(self, current_alignment):
        """Calcula el control PID para velocidad lineal basado en alineación"""
        now = time.time()
        dt = now - self.prev_time_lin
        if dt <= 0:
            dt = 0.1
            
        self.prev_time_lin = now
        error = self.setpoint_lin - current_alignment  # Error de alineación
        
        # Término proporcional
        proportional = self.Kp_lin * error
        
        # Término integral
        self.integral_lin += error * dt
        # Anti-windup: limitar integral
        max_integral = 0.5
        self.integral_lin = max(-max_integral, min(max_integral, self.integral_lin))
        integral = self.Ki_lin * self.integral_lin
        
        # Término derivativo
        derivative = self.Kd_lin * (error - self.last_error_lin) / dt
        self.last_error_lin = error
        
        # Salida PID
        output = proportional + integral + derivative
        
        # Convertir a velocidad lineal válida
        base_speed = self.max_linear_speed * 0.8  # 80% de velocidad máxima como base
        throttle = base_speed * (1.0 - abs(error))  # Reducir velocidad si no está alineado
        
        # Aplicar corrección PID
        throttle += output * 0.1  # Factor de escala para el PID lineal
        
        # Asegurar que no sea negativo y esté en rango
        throttle = max(0.0, min(self.max_linear_speed, throttle))
        
        return throttle

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