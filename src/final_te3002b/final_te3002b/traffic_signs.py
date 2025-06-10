#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from custom_interfaces.srv import SetProcessBool

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

        self.compressed_sub = self.create_subscription(CompressedImage, '/video_source/raw/compressed', self.compressed_callback, 10)
        self.bridge = CvBridge()

        

        # Publicadores
        self.sign_pub = self.create_publisher(String, '/detected_sign', 10)
        self.debug_pub = self.create_publisher(Image, '/yolo_signals_view', 10)

        self.last_detected_sign = None  # Para evitar logs repetitivos

        self.get_logger().info("🧠 Cargando modelo YOLO de señales...")
        self.model = YOLO("/home/navelaz/runs_signs/detect/train/weights/signDetection.pt")
        self.get_logger().info("✅ Modelo YOLO de señales cargado correctamente.")

    def compressed_callback(self, msg):
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
            debug_img_msg.header.frame_id = "yolo_signals_view"
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
