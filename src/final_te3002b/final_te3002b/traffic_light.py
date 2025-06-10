#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from ultralytics import YOLO
from custom_interfaces.srv import SetProcessBool  # ✅ Importa el servicio

class YOLOTester(Node):
    def __init__(self):
        super().__init__('yolo_tester')

        # Estado del sistema
        self.system_running = False

        # QoS para imagen
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.compressed_sub = self.create_subscription(
            CompressedImage, '/video_source/raw/compressed', self.compressed_callback, 10
        )

        self.bridge = CvBridge()

        # Publicadores
        self.sign_pub = self.create_publisher(String, '/detected_color', 10)
        self.debug_pub = self.create_publisher(Image, '/yolo_colors_view', 10)

        self.last_detected_sign = None  # Para evitar logs repetitivos

        self.get_logger().info("🧠 Cargando modelo YOLO de semaforo...")
        self.model = YOLO("/home/navelaz/runs/detect/train/weights/best.pt")
        self.get_logger().info("✅ Modelo YOLO de semaforo cargado correctamente.")

        # ✅ Cliente del servicio
        self.cli = self.create_client(SetProcessBool, 'EnableProcess')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ Esperando el servicio EnableProcess...')

        self.send_request(True)

    def send_request(self, enable):
        request = SetProcessBool.Request()
        request.enable = enable
        future = self.cli.call_async(request)
        future.add_done_callback(self.callback_response)

    def callback_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.system_running = True
                self.get_logger().info("✅ Proceso habilitado por el servicio.")
            else:
                self.system_running = False
                self.get_logger().error("❌ Error al habilitar el proceso.")
                self.get_logger().warn(f'Mensaje: {response.message}')
        except Exception as e:
            self.system_running = False
            self.get_logger().error(f"❌ Excepción al llamar al servicio: {e}")

    def compressed_callback(self, msg):
        if not self.system_running:
            return  # ❌ No hace nada si el sistema aún no está activo

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"❌ Error en compressed_callback: {e}")

    def process_frame(self, frame):
        try:
            results = self.model(frame, verbose=False)[0]

            img_with_boxes = results.plot()
            debug_img_msg = self.bridge.cv2_to_imgmsg(img_with_boxes, encoding="bgr8")
            debug_img_msg.header.stamp = self.get_clock().now().to_msg()
            debug_img_msg.header.frame_id = "yolo_colors_view"
            self.debug_pub.publish(debug_img_msg)

            if results.boxes and results.names:
                max_conf = 0
                detected_class = None
                for box in results.boxes:
                    conf = float(box.conf)
                    class_id = int(box.cls)
                    class_name = results.names[class_id]
                    if conf > max_conf:
                        max_conf = conf
                        detected_class = class_name

                if detected_class:
                    msg = String()
                    msg.data = detected_class
                    self.sign_pub.publish(msg)

                    if detected_class != self.last_detected_sign:
                        self.last_detected_sign = detected_class

        except Exception as e:
            self.get_logger().error(f"❌ Error en process_frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
