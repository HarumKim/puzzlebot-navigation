
import cv2
import numpy as np

class HomographySelector:
    def __init__(self, video_path, output_size=(500, 500)):
        self.video_path = video_path
        self.output_size = output_size
        self.H = None

    def select_points(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            raise ValueError("❌ No se pudo leer el primer frame del video.")

        print("Selecciona 4 puntos en el orden: ⬉ ⬈ ⬊ ⬋")
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"{len(points)}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Selecciona 4 puntos", frame)

        cv2.imshow("Selecciona 4 puntos", frame)
        cv2.setMouseCallback("Selecciona 4 puntos", click_event)

        while True:
            key = cv2.waitKey(1)
            if key == 27 and len(points) == 4:  # ESC
                break
            elif key == 27:
                print("❗ Necesitas seleccionar 4 puntos.")

        cv2.destroyAllWindows()
        cap.release()

        pts_src = np.array(points, dtype=np.float32)
        w, h = self.output_size
        pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        self.H, _ = cv2.findHomography(pts_src, pts_dst)
        print("✅ Matriz de homografía calculada:")
        print(self.H)
        return self.H

    def save_homography(self, filepath="/home/ximena/Downloads/homography_matrix.npy"):
        if self.H is not None:
            np.save(filepath, self.H)
            print(f"✅ Homografía guardada en '{filepath}'")
        else:
            print("❗ Primero necesitas calcular la homografía con select_points().")