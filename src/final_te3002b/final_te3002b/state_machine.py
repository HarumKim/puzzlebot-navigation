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
        self.STATE_1 = 1
        self.STATE_2 = 2
        self.STATE_3 = 3
        self.current_state = self.STATE_1

        # Subscripciones
        self.create_subscription(String, '/line_detected', self.line_callback, 10)
        self.create_subscription(String, '/detected_sign', self.sign_callback, 10)
        self.create_subscription(String, '/detected_color', self.traffic_callback, 10)

        # Publicador de cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Entradas de sensores
        self.line_status = "not_detected"
        self.current_sign = "none"
        self.traffic_light = "none"

        # Flags
        self.pending_dotline = False
        self.dotline_detected = False
        self.stop_handled = False
        self.state_start_time = time.time()
        self.dotline_mode = False  # Se activa cuando se detecta "dotLine"
        self.first_light_detected = False # para evitar múltiples acciones por semáforo
        self.waiting_for_new_color = False  # Se activa cuando se detecta semáforo en rojo

        
        # Timer de ejecución continua
        self.create_timer(0.1, self.state_loop)

        self.get_logger().info("🚦 Máquina de estados iniciada")

    def line_callback(self, msg):
        self.line_status = msg.data

        # Si hay dotLine pendiente y ya no se ve línea → activarlo
        if self.pending_dotline and self.line_status != "detected":
            self.get_logger().info("⚠️ dotLine pendiente activado tras pérdida de línea")
            self.dotline_mode = True
            self.pending_dotline = False
            self.publish_stop()

    def sign_callback(self, msg):
        self.current_sign = msg.data
        self.get_logger().info(f"🚧 Señal recibida: {self.current_sign}")

        if self.current_sign == "dotLine":
            if self.line_status != "detected" and not self.dotline_mode:
                self.get_logger().info("➿ Señal dotLine detectada sin línea → modo semáforo activado")
                self.dotline_mode = True
                self.publish_stop()
            else:
                self.pending_dotline = True

    def traffic_callback(self, msg):
        self.traffic_light = msg.data

    def state_loop(self):
        if self.current_state == self.STATE_1:
            self.handle_state_1()
        elif self.current_state == self.STATE_2:
            self.handle_state_2()
        elif self.current_state == self.STATE_3:
            self.handle_state_3()

    def handle_state_1(self):
        if self.dotline_mode:
            self.get_logger().info("dotline_mode!")
            # 🟥 Si aún no hemos detectado el primer color
            if not self.first_light_detected:
                self.get_logger().info("first_light_detected!")
                if self.traffic_light in ["red", "green", "yellow"]:
                    self.initial_light_color = self.traffic_light
                    self.first_light_detected = True
                    self.get_logger().info(f"🚦 Primer color detectado: {self.initial_light_color}")

                    if self.initial_light_color == "red":
                        self.get_logger().info("🔴 En rojo → esperando nuevo color...")
                        self.waiting_for_new_color = True
                        self.publish_stop()
                    else:
                        if self.initial_light_color == "green":
                            self.get_logger().info("🟢 Semáforo verde → avanzando")
                            self.publish_forward()
                        elif self.initial_light_color == "yellow":
                            self.get_logger().info("🟡 Semáforo amarillo → avanzando lento")
                            self.publish_forward(slow=True)

            # ✅ Ya habíamos detectado rojo, ahora esperamos cambio a verde
            elif self.waiting_for_new_color:
                if self.traffic_light != self.initial_light_color and self.traffic_light == "green":
                    self.get_logger().info("🟢 Cambio a verde → avanzando")
                    self.publish_forward()

                    # Reiniciar flags
                    self.first_light_detected = False
                    self.waiting_for_new_color = False
                    self.initial_light_color = "none"

        # 🔁 Transición al STATE_2
        if self.dotline_mode and self.line_status == "detected":
            self.get_logger().info("🔁 Línea continua detectada → cambiando a STATE_2")
            self.current_state = self.STATE_2
            self.state_start_time = time.time()
            self.dotline_mode = False
            self.first_light_detected = False
            self.initial_light_color = "none"


    def handle_state_2(self):
        # 1️⃣ Transición al estado 3 si detecta línea continua
        if self.line_status == "detected":
            self.get_logger().info("🔁 Línea continua detectada → cambiando a STATE_3")
            self.current_state = self.STATE_3
            self.state_start_time = time.time()
            return

        # 2️⃣ Si NO hay línea y está en modo dotLine, actuar según señal
        if self.dotline_mode:
            self.get_logger().info("⛔ Línea no detectada y modo dotLine activo → detención por seguridad")

            self.publish_stop()

            if self.current_sign == "stop":
                self.get_logger().info("🛑 Señal 'stop' → STOP (confirmado)")
                self.publish_stop()
            elif self.current_sign != "none":
                self.get_logger().info(f"⚠️ Señal '{self.current_sign}' → ignorada temporalmente")

            return

        # 3️⃣ Mientras hay línea pero aún no entra en modo dotLine → obedece señales
        if self.current_sign == "stop":
            self.get_logger().info("🛑 Señal 'stop' detectada")
            self.publish_stop()
        elif self.current_sign != "none":
            self.get_logger().info(f"⚠️ Señal '{self.current_sign}' → ignorada")
        else:
            self.publish_forward()



    def handle_state_3(self):
        if self.current_sign == "stop" and not self.stop_handled:
            self.get_logger().info("🛑 STOP detectado: esperando 3 segundos")
            self.publish_stop()
            time.sleep(3)
            self.publish_stop()
            self.stop_handled = True
        else:
            self.publish_forward()



    #def apply_signal_behavior(self, sign):
    #    if sign == "aheadOnly":
    #        self.publish_forward()
    #    elif sign == "turnLeft":
    #        self.publish_turn(left=True)
    #    elif sign == "turnRight":
    #        self.publish_turn(left=False)
    #    elif sign == "roundabout":
    #        self.publish_forward(slow=True)
    #    elif sign == "stop":
    #        self.publish_stop()
    #    elif sign == "giveWay":
    #       self.get_logger().info("⚠️ Señal 'giveWay': esperando 2s")
    #        self.publish_stop()
    #        time.sleep(5)
    #    elif sign == "roadwork":
    #        self.publish_forward(slow=True)

    def apply_signal_behavior(self, sign):
        if sign == "stop":
            self.publish_stop()
            self.get_logger().info("🛑 Señal STOP: robot detenido.")
        else:
            self.get_logger().info(f"🚫 Señal ignorada: {sign}")


    def publish_forward(self, slow=False):
        self.get_logger().info("FORWARD")
        twist = Twist()
        twist.linear.x = 0.05 if slow else 0.07
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)           # ✅ ¡Publicar ya!
        self.last_twist = twist  # solo actualiza

    def publish_turn(self, left=True):
        twist = Twist()
        twist.linear.x = 0.02
        twist.angular.z = 0.5 if left else -0.5
        self.cmd_pub.publish(twist)           # ✅ ¡Publicar ya!
        self.last_twist = twist

    def publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)           # ✅ ¡Publicar ya!
        self.last_twist = twist


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
