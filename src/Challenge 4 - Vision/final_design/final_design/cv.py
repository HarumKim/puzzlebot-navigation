import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import mediapipe as mp
import numpy as np

class cv(Node):
    def __init__(self):
        super().__init__('cv')
        
        # Creación de publishers
        self.angle_publisher_ = self.create_publisher(Float32, 'angle', 10)
        
        # Timer para procesamiento de frames
        self.timer = self.create_timer(0.1, self.process_frame)

        # Inicialización de MediaPipe para detección de manos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Inicialización de la cámara
        self.cap = cv2.VideoCapture(0)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("No se pudo capturar el cuadro de la cámara.")
            return
        
        # Convertir la imagen a formato RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)
        
        # Inicialización de la variable para el ángulo
        angle = None
        
        # Si se detectan manos, calcular el ángulo de la mano
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x1, y1 = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1]), \
                         int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame.shape[0])
                x2, y2 = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x * frame.shape[1]), \
                         int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * frame.shape[0])
                
                # Calcular el ángulo entre la muñeca y el dedo índice
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if angle < 0:
                    angle += 180
                
                # Publicar el ángulo calculado
                msg_angle = Float32()
                msg_angle.data = angle
                self.angle_publisher_.publish(msg_angle)
                self.get_logger().info(f'Publicado ángulo: {angle:.2f}°')
                
                # Dibujar la línea que representa el ángulo en el cuadro
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Angulo: {int(angle)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Mostrar la ventana con el video
        cv2.imshow('Angle Detection', frame)

        # Cerrar el programa si se presiona 'q'
        if cv2.waitKey(1) == ord('q'):
            self.destroy_node()
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    cv_node = cv()
    try:
        rclpy.spin(cv_node)
    except KeyboardInterrupt:
        pass
    finally:
        cv_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()