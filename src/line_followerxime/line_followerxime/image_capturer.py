#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import cv2
import time

class ImageCapturer(Node):
    def __init__(self):
        super().__init__('image_capturer')

        self.image_counter = 0
        self.max_images = 150
        self.capture_interval = 2.0  # segundos
        self.last_capture_time = time.time()
        self.dataset_path = os.path.expanduser('~/ros2_ws/dataset3')
        os.makedirs(self.dataset_path, exist_ok=True)

        self.bridge = CvBridge()
        self.last_msg_time = None
        self.latest_frame = None
        self.window_opened = False

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Tópico correcto según tu sistema
            self.listener_callback,
            10
        )

        self.get_logger().info('📸 Nodo image_capturer iniciado. Esperando imágenes desde /image_raw...')

    def listener_callback(self, msg):
        current_time = time.time()
        self.last_msg_time = current_time

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame = frame

            # Guardar imagen cada 5 segundos (no mostrar spam en consola)
            if self.image_counter < self.max_images and (current_time - self.last_capture_time) >= self.capture_interval:
                filename = os.path.join(self.dataset_path, f'frame_{self.image_counter:03d}.jpg')
                if cv2.imwrite(filename, frame):
                    height, width = frame.shape[:2]
                    print(f'✅ Imagen {self.image_counter+1}/{self.max_images} guardada ({width}x{height}): {filename}')
                else:
                    print(f'⚠️ Error al guardar imagen: {filename}')
                self.image_counter += 1
                self.last_capture_time = current_time

        except Exception as e:
            self.get_logger().error(f'❌ Error al procesar imagen: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageCapturer()
    try:
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.latest_frame is not None:
                if not node.window_opened:
                    cv2.namedWindow("📷 Vista de cámara (/image_raw)", cv2.WINDOW_NORMAL)
                    node.window_opened = True

                cv2.imshow("📷 Vista de cámara (/image_raw)", node.latest_frame)
                key = cv2.waitKey(20)
                if key == ord('q'):
                    print("⛔ Se presionó 'q'. Cerrando nodo.")
                    break

            if node.last_msg_time is None and (time.time() - start_time) > 5:
                node.get_logger().warn('⚠️ No se han recibido imágenes. ¿Está encendida la cámara?')
                start_time = time.time()

    except KeyboardInterrupt:
        print("\n⛔ Interrupción con Ctrl+C. Cerrando nodo.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
