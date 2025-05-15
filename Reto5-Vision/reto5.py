from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from homografia import HomographySelector
from tracker import MultiPollitoTrackerHybrid
from cameraman import CameraMotionEstimator

def medir_px_por_cm(homography_path, video_path, output_size=(800, 800), mosaico_cm=30):
    H = np.load(homography_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("No se pudo leer el primer frame.")

    frame_warped = cv2.warpPerspective(frame, H, output_size)
    puntos = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 2:
            puntos.append((x, y))
            cv2.circle(frame_warped, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Selecciona extremos de un mosaico", frame_warped)

    print("📐 Selecciona dos puntos que representen un lado del mosaico de 30 cm")
    cv2.imshow("Selecciona extremos de un mosaico", frame_warped)
    cv2.setMouseCallback("Selecciona extremos de un mosaico", click_event)

    while True:
        key = cv2.waitKey(1)
        if key == 27 and len(puntos) == 2:
            break
        elif key == 27:
            print("Selecciona 2 puntos primero.")

    cv2.destroyAllWindows()
    cap.release()

    (x1, y1), (x2, y2) = puntos
    distancia_px = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
    px_to_cm = mosaico_cm / distancia_px
    print(f"📏 {distancia_px:.2f} px ≈ {mosaico_cm} cm → Factor: {px_to_cm:.4f} cm/px")
    return px_to_cm

# -------------------- CONFIGURACIÓN --------------------
model_path = "/home/ximena/runs/detect/train2/weights/best.pt"
video_path = "/home/ximena/Downloads/pollitos.mp4"
CONFIDENCE_THRESHOLD = 0.86
OUTPUT_SIZE = (800, 800)
H_FILE = "/home/ximena/Downloads/homography_matrix.npy"

# -------------------- HOMOGRAFÍA --------------------
if os.path.exists(H_FILE):
    H = np.load(H_FILE)
    px_to_cm = medir_px_por_cm(H_FILE, video_path, output_size=OUTPUT_SIZE)
else:
    selector = HomographySelector(video_path, output_size=OUTPUT_SIZE)
    H = selector.select_points()
    os.makedirs(os.path.dirname(H_FILE), exist_ok=True)
    selector.save_homography(H_FILE)

# -------------------- FUNCIONES ORB --------------------
def extract_keypoints_and_descriptors(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detectAndCompute(gray, None)

def match_keypoints(desc1, desc2, matcher):
    matches = matcher.knnMatch(desc1, desc2, k=2)
    return [m for m, n in matches if m.distance < 0.75 * n.distance]


# -------------------- INICIALIZACIONES --------------------
print("Cargando modelo YOLO...")
model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("No se pudo abrir el video.")

ret, frame = cap.read()
if not ret:
    raise Exception("No se pudo leer el primer frame.")
h, w = frame.shape[:2]

K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
D = np.array([-0.3, 0.1, 0, 0], dtype=np.float32)
new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)

tracker = MultiPollitoTrackerHybrid()
frame_count = 0
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# ORB + matcher
orb = cv2.ORB_create(nfeatures=1000)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
prev_warped, prev_kp, prev_desc = None, None, None
camera_motion = CameraMotionEstimator()

# -------------------- PROCESAMIENTO DE FRAMES --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    tiempo = frame_count / frame_rate

    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(frame, H, OUTPUT_SIZE)
    camera_motion.update(warped)


    # ORB
    kp, desc = extract_keypoints_and_descriptors(warped, orb)

    # YOLO detección
    results = model.predict(warped, imgsz=416, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections = []

    if len(results[0].boxes) > 0:
        for x, y, w_box, h_box in results[0].boxes.xywh.cpu().numpy():
            area = w_box * h_box
            if 500 < area < 5000:
                detections.append(np.array([x, y]))
                top_left = (int(x - w_box / 2), int(y - h_box / 2))
                bottom_right = (int(x + w_box / 2), int(y + h_box / 2))
                cv2.rectangle(warped, top_left, bottom_right, (0, 255, 0), 2)
                cv2.circle(warped, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(warped, "pollito?", (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    tracker.update(detections, tiempo)

    # 🔁 ORB fallback si no hubo detecciones
    if len(detections) == 0 and prev_desc is not None and desc is not None:
        matches = match_keypoints(prev_desc, desc, bf_matcher)
        if len(matches) > 10:
            pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_curr = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, inliers = cv2.estimateAffinePartial2D(pts_prev, pts_curr)

            if M is not None:
                #print(f"[Frame {frame_count}] 🟡 Fallback ORB aplicado con {len(inliers)} puntos.")
                for tid, t in tracker.trackers.items():
                    pred = np.array([[t.trajectory[-1]]], dtype=np.float32)
                    moved = cv2.transform(pred, M)[0, 0]
                    t.update(moved, tiempo)

    cv2.imshow("Tracking Pollitos", warped)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_warped = warped.copy()
    prev_kp, prev_desc = kp, desc

cap.release()
cv2.destroyAllWindows()
print("Procesamiento finalizado.")

# -------------------- RESULTADOS --------------------
pollitos = tracker.get_trajectories()
id_map = {key: f"pollito_{i+1}" for i, key in enumerate(sorted(pollitos.keys()))}
pollitos_renombrados = {id_map[k]: v for k, v in pollitos.items()}

# -------------------- CÁLCULO DE DESPLAZAMIENTO FILTRADO --------------------
MIN_STEP = 2     # Ignorar pequeños movimientos (ruido)
MAX_STEP = 200   # Ignorar saltos irreales (errores de tracking)

desplazamientos = {}
for key, datos in pollitos_renombrados.items():
    total = 0
    for i in range(len(datos['x']) - 1):
        dx = datos['x'][i + 1] - datos['x'][i]
        dy = datos['y'][i + 1] - datos['y'][i]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if MIN_STEP < dist < MAX_STEP:
            total += dist
    desplazamientos[key] = total * px_to_cm  # 🔁 convertir a cm reales


df = pd.DataFrame({
    "Pollito ID": list(desplazamientos.keys()),
    "Desplazamiento (cm)": list(desplazamientos.values())
})
print(df)

# -------------------- MOVIMIENTO DEL CAMARÓGRAFO --------------------
positions, angles = camera_motion.get_trajectory()
x_cam, y_cam = positions[:, 0], positions[:, 1]
# Convertimos a centímetros
x_cam_cm = x_cam * px_to_cm
y_cam_cm = y_cam * px_to_cm

# Calculamos desplazamiento acumulado (trayectoria total recorrida)
cam_total_dist = 0
for i in range(len(x_cam_cm) - 1):
    dx = x_cam_cm[i + 1] - x_cam_cm[i]
    dy = y_cam_cm[i + 1] - y_cam_cm[i]
    dist = np.sqrt(dx**2 + dy**2)
    cam_total_dist += dist

print(f"📸 Desplazamiento total del camarógrafo: {cam_total_dist:.2f} cm")

plt.figure()
plt.plot(x_cam_cm, y_cam_cm, color='black')
plt.title("Movimiento 2D del Camarógrafo (cm)")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.grid(True)
plt.axis("equal")
plt.show()

# Traza 3D (con ángulo como Z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_cam_cm, y_cam_cm, angles, color='blue')
ax.set_title("Trayectoria 3D del Camarógrafo (con giro)")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Rotación (°)")
plt.show()

# -------------------- GRAFICAS DE POLLITOS --------------------
plt.figure()
for key, datos in pollitos_renombrados.items():
    plt.plot(datos['x'], datos['y'], label=key)
plt.title("Trayectorias 2D de los Pollitos")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for key, datos in pollitos_renombrados.items():
    ax.plot(datos['x'], datos['y'], datos['t'], label=key)
ax.set_title("Trayectorias 3D de los Pollitos")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Tiempo (s)")
ax.legend()
plt.show()
