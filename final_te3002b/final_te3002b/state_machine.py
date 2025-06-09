#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        # Estados
        self.FOLLOW_LINE = 1
        self.INTERSECTION = 2
        self.current_state = self.FOLLOW_LINE

        # Subscripciones
        self.create_subscription(String, '/line_detected', self.line_callback, 10)
        self.create_subscription(String, '/detected_sign', self.sign_callback, 10)
        self.create_subscription(String, '/detected_color', self.traffic_callback, 10)

        # Publicador de cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.last_twist = Twist()

        # Variables internas
        self.line_status = "not_detected"
        self.current_sign = "none"
        self.traffic_light = "none"

        self.dotline_detected = False
        self.selected_intersection_sign = "none"
        self.last_valid_sign = "none"

        self.first_light_detected = False
        self.waiting_for_new_color = False
        self.initial_light_color = "none"

        self.continuous_forward = False

        # Timer principal
        self.create_timer(0.1, self.state_loop)
        self.get_logger().info("🚦 Máquina de estados iniciada")

    def line_callback(self, msg):
        self.line_status = msg.data

        if self.current_state == self.FOLLOW_LINE:
            if self.line_status == "detected":
                self.publish_forward()
            else:
                self.publish_stop()

        elif self.current_state == self.INTERSECTION and self.line_status == "detected":
            self.get_logger().info("➡️ Línea detectada → volviendo a FOLLOW_LINE")
            self.reset_intersection_flags()
            self.current_state = self.FOLLOW_LINE

    def sign_callback(self, msg):
        self.current_sign = msg.data

        if self.current_sign == "dotLine" and self.current_state == self.FOLLOW_LINE:
            if self.line_status != "detected":
                self.get_logger().info("⚠️ dotLine detectado y línea perdida → entrando a INTERSECTION")
                self.dotline_detected = True
                self.selected_intersection_sign = self.last_valid_sign  # usar la última señal válida
                self.current_state = self.INTERSECTION
                self.publish_stop()
        elif self.current_sign not in ["none", "dotLine", "red", "green", "yellow"]:
            self.last_valid_sign = self.current_sign  # solo guardar señales útiles
            if self.current_state == self.INTERSECTION:
                self.selected_intersection_sign = self.current_sign

    def traffic_callback(self, msg):
        self.traffic_light = msg.data

    def state_loop(self):
        if self.current_state == self.FOLLOW_LINE:
            self.handle_follow_line()
        elif self.current_state == self.INTERSECTION:
            self.handle_intersection()

        if self.continuous_forward:
            self.cmd_pub.publish(self.last_twist)

    def handle_follow_line(self):
        if self.line_status == "detected":
            self.publish_forward()
            if self.current_sign not in ["none", "dotLine", "red", "green", "yellow"]:
                self.apply_signal_behavior(self.current_sign)

    def handle_intersection(self):
        if not self.first_light_detected:
            if self.traffic_light in ["red", "green", "yellow"]:
                self.initial_light_color = self.traffic_light
                self.first_light_detected = True
                self.get_logger().info(f"🚦 Color inicial en intersección: {self.initial_light_color}")

                if self.initial_light_color == "red":
                    self.get_logger().info("🔴 En rojo → esperando nuevo color...")
                    self.waiting_for_new_color = True
                    self.publish_stop()
                else:
                    if self.initial_light_color == "green":
                        self.get_logger().info("🟢 Semáforo verde → ejecutando acción")
                        self.execute_action_from_sign(slow=False)
                    elif self.initial_light_color == "yellow":
                        self.get_logger().info("🟡 Semáforo amarillo → ejecutando acción lenta")
                        self.execute_action_from_sign(slow=True)

        elif self.waiting_for_new_color:
            if self.traffic_light == "green":
                self.get_logger().info("🟢 Cambio a verde → ejecutando acción")
                self.waiting_for_new_color = False
                self.execute_action_from_sign(slow=False)

    def execute_action_from_sign(self, slow=False):
        if self.selected_intersection_sign != "none":
            self.get_logger().info(f"🚗 Ejecutando acción: {self.selected_intersection_sign}")
            self.apply_signal_behavior(self.selected_intersection_sign, slow=slow)
            self.continuous_forward = True

    def apply_signal_behavior(self, sign, slow=False):
        if sign == "aheadOnly":
            self.get_logger().info(f"⬆️ Señal 'aheadOnly' → avanzar {'lento' if slow else 'rápido'}")
            self.publish_forward(slow=slow)
        elif sign == "turnLeft":
            self.get_logger().info("⬅️ Señal 'turnLeft' → girar a la izquierda")
           # self.publish_turn(left=True)
        elif sign == "turnRight":
            self.get_logger().info("➡️ Señal 'turnRight' → girar a la derecha")
            #self.publish_turn(left=False)
        elif sign == "stop":
            self.get_logger().info("🛑 Señal 'stop' → detenerse")
            self.publish_stop()
        elif sign == "giveWay":
            self.get_logger().info("⚠️ Señal 'giveWay' → ceder paso 2s")
            self.publish_stop()
            time.sleep(2)
        elif sign == "roadwork":
            self.get_logger().info("🚧 Señal 'roadwork' → avanzar lento")
            #self.publish_forward(slow=True)
        else:
            self.get_logger().info(f"🔶 Señal desconocida o ignorada: {sign}")

    def publish_forward(self, slow=False):
        twist = Twist()
        twist.linear.x = 0.06 if slow else 0.08
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.last_twist = twist

    def publish_turn(self, left=True):
        twist = Twist()
        twist.linear.x = 0.02
        twist.angular.z = 0.5 if left else -0.5
        self.cmd_pub.publish(twist)
        self.last_twist = twist

    def publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.last_twist = twist

    def reset_intersection_flags(self):
        self.dotline_detected = False
        self.first_light_detected = False
        self.waiting_for_new_color = False
        self.initial_light_color = "none"
        self.selected_intersection_sign = "none"
        self.continuous_forward = False


def main(args=None):
    rclpy.init(args=args)
    node = StateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()