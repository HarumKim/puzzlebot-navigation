#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from ultralytics import YOLO
from custom_interfaces.srv import SetProcessBool

class YOLOTester(Node):
    def __init__(self):
        super().__init__('yolo_tester')

        # Estado de sistema
        self.system_running = False

        # QoS para imagen
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos_profile)
        self.bridge = CvBridge()

        # Publicador de señal detectada
        self.sign_pub = self.create_publisher(String, '/detected_color', 10)

        self.last_detected_color = None  # Guarda el último color detectado

        # Inicializar modelo YOLO
        self.get_logger().info("🧠 Cargando modelo YOLO de semaforo...")
        self.model = YOLO("/home/navelaz/runs/detect/train/weights/best.pt")
        self.get_logger().info("✅ Modelo YOLO de semaforo cargado correctamente.")

        cv2.namedWindow("YOLO Colors", cv2.WINDOW_NORMAL)

        # Cliente de servicio
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

    def image_callback(self, msg):
        if not self.system_running:
            return  # No hacer nada si el sistema no está habilitado

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Inferencia YOLO
            results = self.model(frame, verbose=False)[0]
            img_with_boxes = results.plot()
            cv2.imshow("YOLO Colors", img_with_boxes)
            cv2.waitKey(1)

            # Procesar detecciones
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

                    if detected_class != self.last_detected_color:
                        #self.get_logger().info(f"🚦 Nuevo color detectado: {detected_class}")
                        self.last_detected_color = detected_class

        except Exception as e:
            self.get_logger().error(f"❌ Error en la inferencia YOLO: {e}")

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
