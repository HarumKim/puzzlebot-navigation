#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import time

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        self.FOLLOW_LINE = 1
        self.INTERSECTION = 2
        self.current_state = self.FOLLOW_LINE

        self.create_subscription(String, '/line_detected', self.line_callback, 10)
        self.create_subscription(String, '/detected_sign', self.sign_callback, 10)
        self.create_subscription(String, '/detected_color', self.traffic_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.last_twist = Twist()

        self.line_status = "not_detected"
        self.current_sign = "none"
        self.traffic_light = "none"
        self.turning_in_progress = False

        self.dotline_detected = False
        self.selected_intersection_sign = "none"
        self.last_valid_sign = "none"

        self.first_light_detected = False
        self.waiting_for_new_color = False
        self.initial_light_color = "none"

        self.continuous_forward = False
        # Speed factor for velocity control
        self.speed_factor_pub = self.create_publisher(Float32, '/velocity_factor', 10)
        self.current_speed_factor = 1.0
        self.last_roadwork_time = self.get_clock().now().seconds_nanoseconds()[0]

        #variables for roadwork detection
        self.roadwork_active = False

        # Variables para giveWay
        self.giveway_active = False
        self.giveway_waiting = False
        self.giveway_wait_start = 0.0


        # Variables para stop
        self.stop_waiting = False
        self.stop_wait_start = 0.0

        self.create_timer(0.1, self.state_loop)
        self.get_logger().info("🚦 Máquina de estados iniciada")

    def line_callback(self, msg):
        self.line_status = msg.data
        if self.current_state == self.INTERSECTION:
            if self.turning_in_progress:
                return  # Ignorar detección de línea si está girando
            if self.line_status == "detected":
                self.get_logger().info("➡️ Línea detectada → volviendo a FOLLOW_LINE")
                self.reset_intersection_flags()
                self.current_state = self.FOLLOW_LINE

    def sign_callback(self, msg):
        previous_sign = self.current_sign
        self.current_sign = msg.data

        # Activar reducción de velocidad
        if self.current_sign == "roadwork":
            if not self.roadwork_active:
                self.get_logger().info("🚧 Iniciando zona roadwork → reduciendo velocidad")
                self.update_speed_factor(0.6)
                self.roadwork_active = True
            self.last_roadwork_time = self.get_clock().now().seconds_nanoseconds()[0]

        elif self.current_sign == "stop":
            if not self.stop_waiting:
                self.get_logger().info("🛑 Señal 'stop' detectada → iniciando espera de 8 segundos")
                self.stop_waiting = True
                self.stop_wait_start = self.get_clock().now().seconds_nanoseconds()[0]


        # Activar giveWay
        elif self.current_sign == "giveWay":
            if not self.giveway_active:
                self.get_logger().info("⚠️ Señal 'giveWay' detectada → reduciendo velocidad")
                self.update_speed_factor(0.4)  # Reducir más que roadwork
                self.giveway_active = True

        # Salir de zona roadwork
        elif self.roadwork_active and self.current_sign != "roadwork":
            self.get_logger().info("✅ Fin de zona roadwork → restaurando velocidad")
            self.update_speed_factor(1.0)
            self.roadwork_active = False

        # Resto de señales
        if self.current_sign == "dotLine" and self.current_state == self.FOLLOW_LINE:
            if self.line_status != "detected":
                self.get_logger().info("⚠️ dotLine detectado y línea perdida → entrando a INTERSECTION")
                self.dotline_detected = True
                self.selected_intersection_sign = self.last_valid_sign
                self.current_state = self.INTERSECTION
                self.publish_stop()
        elif self.current_sign not in ["none", "dotLine", "red", "green", "yellow"]:
            self.last_valid_sign = self.current_sign
            if self.current_state == self.INTERSECTION:
                self.selected_intersection_sign = self.current_sign

    def traffic_callback(self, msg):
        self.traffic_light = msg.data

    def update_speed_factor(self, factor: float):
        self.current_speed_factor = factor
        self.speed_factor_pub.publish(Float32(data=factor))

    
    def state_loop(self):
        if self.current_state == self.FOLLOW_LINE:
            self.handle_follow_line()
        elif self.current_state == self.INTERSECTION:
            self.handle_intersection()
        if self.continuous_forward:
            self.cmd_pub.publish(self.last_twist)

            
    def handle_follow_line(self):
        # Ya no forzar aquí el update del factor, solo actuar si fue recibido en sign_callback
        now_sec = self.get_clock().now().seconds_nanoseconds()[0]
        if self.roadwork_active and now_sec - self.last_roadwork_time > 2:
            self.get_logger().info("✅ Expiró señal roadwork → restaurando velocidad")
            self.update_speed_factor(1.0)
            self.roadwork_active = False

        # Manejar espera de stop
        if self.stop_waiting:
            now_sec = self.get_clock().now().seconds_nanoseconds()[0]
            if now_sec - self.stop_wait_start >= 6.0:
                self.get_logger().info("🛑 STOP TOTAL → aplicando velocidad 0.0")
                self.stop_waiting = False
                self.update_speed_factor(0.0)

    def handle_intersection(self):
        # Si giveWay está activo y no hemos empezado a esperar, iniciar espera de 2 segundos
        if self.giveway_active and not self.giveway_waiting and not self.first_light_detected:
            self.get_logger().info("⚠️ En intersección con giveWay → esperando 2 segundos")
            self.giveway_waiting = True
            self.giveway_wait_start = self.get_clock().now().seconds_nanoseconds()[0]
            self.publish_stop()
            return

        # Si estamos esperando por giveWay, verificar si han pasado 2 segundos
        if self.giveway_waiting:
            now_sec = self.get_clock().now().seconds_nanoseconds()[0]
            if now_sec - self.giveway_wait_start >= 2.0:
                self.get_logger().info("✅ Espera de giveWay completada → continuando con lógica normal")
                self.giveway_waiting = False
                # Restaurar velocidad normal después de la espera en intersección
                self.update_speed_factor(1.0)
                self.giveway_active = False
                # Continuar con la lógica normal de intersección
            else:
                # Seguir esperando
                self.publish_stop()
                return

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
                    slow = self.initial_light_color == "yellow"
                    self.execute_action_from_sign(slow=slow)

        elif self.waiting_for_new_color and self.traffic_light == "green":
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
        elif sign == "turnRight":
            self.get_logger().info(f"➡️ Señal 'turnRight' → girar a la derecha {'lento' if slow else 'rápido'}")
            self.publish_turn(left=False, slow=slow)
        elif sign == "turnLeft":
            self.get_logger().info(f"⬅️ Señal 'turnLeft' → girar a la izquierda {'lento' if slow else 'rápido'}")
            self.publish_turn(left=True, slow=slow)
        elif sign == "stop":
            self.get_logger().info("🛑 Señal 'stop' → detenerse")
            self.publish_stop()
        #elif sign == "giveWay":
            #self.get_logger().info("⚠️ Señal 'giveWay' → ceder paso 2s")
            #self.publish_stop()
            #time.sleep(2)
        else:
            self.get_logger().info(f"🔶 Señal desconocida o ignorada: {sign}")

    def publish_forward(self, slow=False):
        twist = Twist()
        twist.linear.x = 0.06 if slow else 0.08
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.last_twist = twist

    def publish_turn(self, left=True, slow=True):
        self.turning_in_progress = True  # ← Inicia flag

        twist = Twist()
        twist.linear.x = 0.06 if slow else 0.08
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.get_logger().info("⏩ Avanzando recto antes de girar")
        time.sleep(4.4 if slow else 3.3)

        twist.linear.x = 0.02
        twist.angular.z = 0.5 if left else -0.5
        self.cmd_pub.publish(twist)
        self.get_logger().info("↪️ Girando a la " + ("izquierda" if left else "derecha"))
        time.sleep(2.5 if slow else 2.5)

        twist.linear.x = 0.06 if slow else 0.08
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.get_logger().info("⏩ Avanzando tras el giro")
        time.sleep(4.0 if slow else 3.0)

        self.turning_in_progress = False  # ← Finaliza flag
        self.get_logger().info("✅ Giro completado")

    def publish_stop(self):
        self.get_logger().info("STOP")
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
        # Reset giveWay flags cuando salimos de intersección
        self.giveway_waiting = False
        if self.giveway_active:
            self.update_speed_factor(1.0)
            self.giveway_active = False

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