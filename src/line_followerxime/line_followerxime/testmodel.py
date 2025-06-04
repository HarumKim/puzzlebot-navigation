#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from ultralytics import YOLO  # Asegúrate de tener ultralytics instalado: pip install ultralytics

class YOLOTester(Node):
    def __init__(self):
        super().__init__('yolo_tester')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos_profile)
        self.bridge = CvBridge()

        # NUEVO: Publisher para señales detectadas
        self.signal_pub = self.create_publisher(String, '/yolo_signal', 10)


        self.get_logger().info("Cargando modelo YOLO...")
        self.model = YOLO("/home/mateo/ros2_ws/src/line_followerxime/signDetection.pt")
        self.get_logger().info("Modelo YOLO cargado correctamente.")

        cv2.namedWindow("YOLO Detecciones", cv2.WINDOW_NORMAL)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Inferencia YOLO
            results = self.model(frame, verbose=False)[0]

            # Obtener la imagen con las detecciones dibujadas
            # results.plot() devuelve una imagen numpy con las cajas, etiquetas, etc
            img_with_boxes = results.plot()

            cv2.imshow("YOLO Detecciones", img_with_boxes)
            cv2.waitKey(1)

            # NUEVO: publicar la primera etiqueta detectada (si hay alguna)
            if results.names and results.boxes.cls.numel() > 0:
                class_id = int(results.boxes.cls[0])
                class_name = results.names[class_id]
                self.signal_pub.publish(String(data=class_name))
        except Exception as e:
            self.get_logger().error(f"Error en la inferencia YOLO: {e}")

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
