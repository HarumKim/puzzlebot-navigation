#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        self.FOLLOW_LINE = 1
        self.INTERSECTION = 2
        self.current_state = self.FOLLOW_LINE

        self.create_subscription(String, '/line_detected', self.line_callback, 10)
        self.create_subscription(String, '/detected_sign', self.sign_callback, 10)
        self.create_subscription(String, '/detected_color', self.traffic_callback, 10)

        self.buzzer_pub = self.create_publisher(String, '/play_tone', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.last_twist = Twist()

        self.line_status = "not_detected"
        self.raw_line_status = "not_detected"  # Nuevo: estado crudo de la línea
        self.current_sign = "none"
        self.traffic_light = "none"
        self.turning_in_progress = False

        self.dotline_detected = False
        self.selected_intersection_sign = "none"
        self.last_valid_sign = "none"
        
        self.stop_seen_once = False
        self.first_light_detected = False
        self.waiting_for_new_color = False
        self.initial_light_color = "none"

        self.continuous_forward = False
        self.speed_factor_pub = self.create_publisher(Float32, '/velocity_factor', 10)
        self.current_speed_factor = 1.0
        self.last_roadwork_time = self.get_clock().now().seconds_nanoseconds()[0]

        # Variables for roadwork detection
        self.roadwork_active = False

        # Variables para giveWay
        self.giveway_active = False
        self.giveway_waiting = False
        self.giveway_wait_start = 0.0

        # Variables para stop
        self.stop_waiting = False
        self.stop_wait_start = 0.0

        # ===== VARIABLES PARA TIMER DE GIRO =====
        self.turn_in_progress = False
        self.turn_phase = 0  # 0: avanzar recto, 1: girar, 2: avanzar final
        self.turn_start_time = 0.0
        self.turn_left = True
        self.turn_slow = False
        
        # Duraciones para cada fase del giro
        self.FORWARD_BEFORE_TURN_SLOW = 4.0 # antes 3.8
        self.FORWARD_BEFORE_TURN_FAST = 3.0 # antes 3.3
        self.TURN_DURATION_SLOW = 2.0 
        self.TURN_DURATION_FAST = 2.0 # antes 2.5
        self.FORWARD_AFTER_TURN_SLOW = 4.0
        self.FORWARD_AFTER_TURN_FAST = 3.0

        self.create_timer(0.1, self.state_loop)
        self.get_logger().info("🚦 Máquina de estados iniciada")

    def line_callback(self, msg):
        # Guardar el estado crudo de la línea
        self.raw_line_status = msg.data
        
        # Actualizar line_status basado en el estado del giro
        self.update_line_status()

    def update_line_status(self):
        """Actualiza line_status basado en el estado del giro y el estado crudo"""
        # Durante las fases 0 y 1 del giro, forzar line_status a "not_detected"
        if self.turn_in_progress and self.turn_phase in [0, 1]:
            self.line_status = "not_detected"
            self.get_logger().debug(f"🚫 Forzando línea no detectada durante fase {self.turn_phase} del giro")
        else:
            # En fase 2 (avance final) y fuera del giro, usar detección normal
            self.line_status = self.raw_line_status
        
        # Manejar transición de estado solo si no estamos en fases restrictivas del giro
        if self.current_state == self.INTERSECTION:
            if self.turning_in_progress or self.turn_in_progress:
                return  # Ignorar cambios de estado si está girando
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
                self.update_speed_factor(0.4)
                self.roadwork_active = True
                self.buzzer_pub.publish(String(data="roadwork"))
            self.last_roadwork_time = self.get_clock().now().seconds_nanoseconds()[0]

        elif self.current_sign == "stop":
            if self.stop_seen_once:
                if not self.stop_waiting:
                    self.get_logger().info("🛑 Confirmación de 'stop' → iniciando espera de 6 segundos")
                    self.buzzer_pub.publish(String(data="stop"))
                    self.stop_waiting = True
                    self.stop_wait_start = self.get_clock().now().seconds_nanoseconds()[0]
                    self.stop_seen_once = False
            else:
                self.get_logger().info("⚠️ Primera detección de 'stop' → esperando confirmación")
                self.stop_seen_once = True

        # Activar giveWay
        elif self.current_sign == "giveWay":
            if not self.giveway_active:
                self.get_logger().info("⚠️ Señal 'giveWay' detectada → reduciendo velocidad")
                self.update_speed_factor(0.4)
                self.giveway_active = True
                self.buzzer_pub.publish(String(data="giveWay"))

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
            if self.current_sign != "stop":
                self.stop_seen_once = False
            elif self.current_state == self.INTERSECTION:
                self.selected_intersection_sign = self.current_sign

    def traffic_callback(self, msg):
        self.traffic_light = msg.data

    def update_speed_factor(self, factor: float):
        self.current_speed_factor = factor
        self.speed_factor_pub.publish(Float32(data=factor))

    def state_loop(self):
        # ===== ACTUALIZAR ESTADO DE LÍNEA EN CADA CICLO =====
        self.update_line_status()
        
        if self.current_state == self.FOLLOW_LINE:
            self.handle_follow_line()
        elif self.current_state == self.INTERSECTION:
            self.handle_intersection()
        
        # ===== MANEJAR TIMER DE GIRO =====
        if self.turn_in_progress:
            self.handle_turn_timer()
            
        if self.continuous_forward:
            self.cmd_pub.publish(self.last_twist)

    def handle_follow_line(self):
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
                self.update_speed_factor(1.0)
                self.giveway_active = False
            else:
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
            self.buzzer_pub.publish(String(data=sign)) 

        elif sign == "turnRight":
            
            self.get_logger().info(f"➡️ Señal 'turnRight' → girar a la derecha {'lento' if slow else 'rápido'}")
            self.start_turn_timer(left=False, slow=slow)
            self.buzzer_pub.publish(String(data=sign)) 
        elif sign == "turnLeft":
            self.get_logger().info(f"⬅️ Señal 'turnLeft' → girar a la izquierda {'lento' if slow else 'rápido'}")
            self.start_turn_timer(left=True, slow=slow)
            self.buzzer_pub.publish(String(data=sign)) 
        else:
            self.get_logger().info(f"🔶 Señal desconocida o ignorada: {sign}")

    def publish_forward(self, slow=False):
        twist = Twist()
        twist.linear.x = 0.06 if slow else 0.08
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.last_twist = twist

    # ===== MÉTODOS PARA TIMER DE GIRO =====
    def start_turn_timer(self, left=True, slow=True):
        """Inicia la secuencia de giro con timer personalizado"""
        self.turning_in_progress = True
        self.turn_in_progress = True
        self.turn_phase = 0
        self.turn_left = left
        self.turn_slow = slow
        self.turn_start_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        # Iniciar primera fase: avanzar recto
        twist = Twist()
        twist.linear.x = 0.06 if slow else 0.08
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.last_twist = twist
        self.get_logger().info("⏩ Fase 0: Avanzando recto antes de girar")

    def handle_turn_timer(self):
        """Maneja las fases del giro usando timer personalizado"""
        now_sec = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed_time = now_sec - self.turn_start_time

        if self.turn_phase == 0:  # Fase: avanzar recto antes de girar
            duration = self.FORWARD_BEFORE_TURN_SLOW if self.turn_slow else self.FORWARD_BEFORE_TURN_FAST
            if elapsed_time >= duration:
                self.turn_phase = 1
                self.turn_start_time = now_sec
                
                # Iniciar giro
                self.line_status = "not_detected"
                twist = Twist()
                twist.linear.x = 0.02
                twist.angular.z = 0.5 if self.turn_left else -0.5
                self.cmd_pub.publish(twist)
                self.last_twist = twist
                self.get_logger().info("↪️ Fase 1: Girando a la " + ("izquierda" if self.turn_left else "derecha") )

        elif self.turn_phase == 1:  # Fase: girando
            duration = self.TURN_DURATION_SLOW if self.turn_slow else self.TURN_DURATION_FAST
            if elapsed_time >= duration:
                self.turn_phase = 2
                self.turn_start_time = now_sec
                
                # Avanzar después del giro
                twist = Twist()
                twist.linear.x = 0.06 if self.turn_slow else 0.08
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                self.last_twist = twist
                self.get_logger().info("⏩ Fase 2: Avanzando tras el giro ")

        elif self.turn_phase == 2:  # Fase: avanzar después del giro
            duration = self.FORWARD_AFTER_TURN_SLOW if self.turn_slow else self.FORWARD_AFTER_TURN_FAST
            if elapsed_time >= duration:
                # Giro completado
                self.finish_turn()

    def finish_turn(self):
        """Finaliza la secuencia de giro"""
        self.turn_in_progress = False
        self.turning_in_progress = False
        self.turn_phase = 0
        
        # Asegurar que el último comando se guarde en last_twist
        twist = Twist()
        twist.linear.x = 0.06 if self.turn_slow else 0.08
        twist.angular.z = 0.0
        self.last_twist = twist
        self.cmd_pub.publish(twist)
        
        self.get_logger().info("✅ Giro completado (línea completamente detectable)")

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
            
        # Reset turn flags
        self.turn_in_progress = False
        self.turning_in_progress = False
        self.turn_phase = 0

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