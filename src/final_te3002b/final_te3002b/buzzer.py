#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import Jetson.GPIO as GPIO
import time

class BuzzerNode(Node):
    def __init__(self):
        super().__init__('buzzer_node')

        self.buzzer_pin = 12
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.buzzer_pin, 440)
        self.pwm.start(0)

        self.subscription = self.create_subscription(String, '/play_tone', self.play_tone_callback, 10)

        self.melodies = {
            "turnRight": [(523, 0.15), (587, 0.15), (659, 0.2)],  # C5 → D5 → E5
            "turnLeft": [(659, 0.15), (587, 0.15), (523, 0.2)],   # E5 → D5 → C5
            "roadwork": [(440, 0.25), (392, 0.25), (349, 0.3)],   # A4 → G4 → F4
            "giveWay": [(440, 0.3), (0, 0.1), (466, 0.3)],        # A4 → silencio → Bb4
            "stop": [  # Estrellita Mario Kart (versión corta)
                (659, 0.2), (659, 0.2), (0, 0.1), (659, 0.2),
                (523, 0.2), (659, 0.2), (784, 0.4), (392, 0.4)
            ],
            "aheadOnly": [(440, 0.2), (0, 0.05), (440, 0.2)]

        }

    def play_tone_callback(self, msg):
        melody = self.melodies.get(msg.data, [])
        for freq, duration in melody:
            if freq <= 0:
                self.pwm.ChangeDutyCycle(0)
            else:
                self.pwm.ChangeDutyCycle(50)
                try:
                    self.pwm.ChangeFrequency(freq)
                except OSError as e:
                    self.get_logger().error(f"❌ PWM Error (freq={freq}): {e}")
            time.sleep(duration)
        self.pwm.ChangeDutyCycle(0)

    def destroy_node(self):
        self.pwm.stop()
        GPIO.cleanup()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BuzzerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
