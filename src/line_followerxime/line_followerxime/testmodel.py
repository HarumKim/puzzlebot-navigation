#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from ultralytics import YOLO

class YOLOTester(Node):
    def __init__(self):
        super().__init__('yolo_tester')

        # QoS para imagen
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos_profile)
        self.bridge = CvBridge()

        # Publicador de señal detectada
        self.sign_pub = self.create_publisher(String, '/detected_sign', 10)

        self.get_logger().info("🧠 Cargando modelo YOLO...")
        self.model = YOLO("/home/navelaz/runs/detect/train/weights/signDetection.pt")
        self.get_logger().info("✅ Modelo YOLO cargado correctamente.")

        cv2.namedWindow("YOLO Detecciones", cv2.WINDOW_NORMAL)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Inferencia YOLO
            results = self.model(frame)[0]
            img_with_boxes = results.plot()
            cv2.imshow("YOLO Detecciones", img_with_boxes)
            cv2.waitKey(1)

            # Procesar detecciones
            if results.boxes and results.names:
                # Tomar el nombre de la detección con mayor confianza
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
                    self.get_logger().info(f"🚧 Señal detectada: {detected_class}")

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
