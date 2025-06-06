#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

def nothing(x):
    pass

class HSVCalibrator(Node):
    def __init__(self):
        super().__init__('hsv_calibrator')
        self.bridge = CvBridge()
        self.img = None

        self.subscription = self.create_subscription(
            Image, '/video_source/raw', self.image_callback, 10)

        # Crea ventana con sliders
        cv.namedWindow("Trackbars")
        cv.createTrackbar("H min", "Trackbars", 0, 179, nothing)
        cv.createTrackbar("H max", "Trackbars", 30, 179, nothing)
        cv.createTrackbar("S min", "Trackbars", 100, 255, nothing)
        cv.createTrackbar("S max", "Trackbars", 255, 255, nothing)
        cv.createTrackbar("V min", "Trackbars", 100, 255, nothing)
        cv.createTrackbar("V max", "Trackbars", 255, 255, nothing)

        self.timer = self.create_timer(0.1, self.display)

    def image_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error al convertir imagen: {e}")

    def display(self):
        if self.img is None:
            return

        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        h_min = cv.getTrackbarPos("H min", "Trackbars")
        h_max = cv.getTrackbarPos("H max", "Trackbars")
        s_min = cv.getTrackbarPos("S min", "Trackbars")
        s_max = cv.getTrackbarPos("S max", "Trackbars")
        v_min = cv.getTrackbarPos("V min", "Trackbars")
        v_max = cv.getTrackbarPos("V max", "Trackbars")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(self.img, self.img, mask=mask)

        cv.imshow("Original", self.img)
        cv.imshow("Mascara", mask)
        cv.imshow("Resultado", result)

        # Imprime valores para copiar fácilmente
        print(f"HSV LOWER: {lower} | HSV UPPER: {upper}", end="\r")

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = HSVCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
