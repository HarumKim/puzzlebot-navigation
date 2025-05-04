import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

class ServoVisualizer(Node):
    def __init__(self):
        super().__init__('servo_visualizer')
        self.subscription = self.create_subscription(
            Float32,
            'angle',
            self.listener_callback,
            10)
        
        # Initial angle value
        self.angle = 0.0

        # Launch matplotlib plot in separate thread
        self.plot_thread = Thread(target=self.run_plot, daemon=True)
        self.plot_thread.start()

    def listener_callback(self, msg):
        self.angle = msg.data
        self.get_logger().info(f"Recibido ángulo: {self.angle:.2f}°")

    def run_plot(self):
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        line, = ax.plot([], [], 'o-', lw=3)

        while rclpy.ok():
            # Convert angle to radians and plot servo arm
            theta = np.radians(self.angle)
            x = [0, np.cos(theta)]
            y = [0, np.sin(theta)]
            line.set_data(x, y)

            ax.set_title(f"Servo Angle: {self.angle:.1f}°")
            fig.canvas.draw()
            fig.canvas.flush_events()
            rclpy.spin_once(self, timeout_sec=0.01)

def main(args=None):
    rclpy.init(args=args)
    servo_node = ServoVisualizer()
    try:
        rclpy.spin(servo_node)
    except KeyboardInterrupt:
        pass
    finally:
        servo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
