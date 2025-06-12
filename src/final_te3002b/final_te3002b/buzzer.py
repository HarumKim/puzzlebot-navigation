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
        
        # Melodías actualizadas con las del script
        self.melodies = {
            "turnRight": [(523, 0.15), (587, 0.15), (659, 0.2)],  # C5 → D5 → E5
            "turnLeft": [(659, 0.15), (587, 0.15), (523, 0.2)],   # E5 → D5 → C5
            "roadwork": [(262, 0.08), (196, 0.12), (165, 0.15)],  # Stomp Goomba Sound: C4 → G3 → E3
            "giveWay": [  # Level Down Super Mario Bros Wii
                (392, 0.3), (330, 0.3), (262, 0.4), (196, 0.5), 
                (0, 0.2), (175, 0.4), (147, 0.6), (131, 0.8)
                # G4 → E4 → C4 → G3 → PAUSA → F3 → D3 → C3
            ],
            "stop": [  # Level Complete Super Mario Bros
                (392, 0.15), (523, 0.15), (659, 0.15), (784, 0.15), 
                (1047, 0.15), (1319, 0.15), (1568, 0.4), (0, 0.1),
                (1319, 0.2), (0, 0.05), (659, 0.2), (0, 0.05), (523, 0.6)
                # G4 → C5 → E5 → G5 → C6 → E6 → G6 → PAUSA → E6 → PAUSA → E5 → PAUSA → C5
            ],
            "aheadOnly": [(440, 0.2), (0, 0.05), (440, 0.2)]  # A4 → PAUSA → A4
        }
        
        self.get_logger().info('🎵 BuzzerNode iniciado con melodías de Super Mario')
    
    def play_tone_callback(self, msg):
        melody_name = msg.data
        melody = self.melodies.get(melody_name, [])
        
        if not melody:
            self.get_logger().warn(f"❌ Melodía '{melody_name}' no encontrada")
            return
            
        self.get_logger().info(f"🎵 Reproduciendo: {melody_name}")
        
        for freq, duration in melody:
            if freq <= 0:
                # Pausa/silencio
                self.pwm.ChangeDutyCycle(0)
            else:
                self.pwm.ChangeDutyCycle(50)
                try:
                    self.pwm.ChangeFrequency(freq)
                except OSError as e:
                    self.get_logger().error(f"❌ PWM Error (freq={freq}): {e}")
                    continue
            
            time.sleep(duration)
            # Pequeña pausa entre notas para mejor separación
            self.pwm.ChangeDutyCycle(0)
            time.sleep(0.02)
        
        # Asegurar que el sonido se apague al final
        self.pwm.ChangeDutyCycle(0)
        self.get_logger().info(f"✅ {melody_name} completada")
    
    def destroy_node(self):
        self.get_logger().info('🔇 Deteniendo BuzzerNode...')
        self.pwm.stop()
        GPIO.cleanup()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BuzzerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Interrupción detectada")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("🎼 BuzzerNode terminado")

if __name__ == '__main__':
    main()
