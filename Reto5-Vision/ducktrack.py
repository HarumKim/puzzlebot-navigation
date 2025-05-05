import cv2
import numpy as np
from ultralytics import YOLO
from motpy import Detection, MultiObjectTracker
import os

# === CONFIGURACIÓN ===
VIDEO_PATH = "pollitos.mp4"
MODEL_PATH = r"C:\Users\aloch\runs\detect\train2\weights\best.pt" 
TARGET_CLASS_NAME = "ducks"  # Solo si usas modelo entrenado con patitos
MAX_TRACKS = 100
colors = np.random.randint(0, 255, size=(MAX_TRACKS, 3), dtype="uint8")

# === CARGA MODELO YOLOv8 LOCAL ===

model = YOLO(MODEL_PATH)

# === TRACKING CON MOTPY ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
multi_tracker = MultiObjectTracker(
    dt=1/fps,
    tracker_kwargs={'max_staleness': 5},
    model_spec={
        'order_pos': 1, 'dim_pos': 2,
        'order_size': 0, 'dim_size': 2,
        'q_var_pos': 5000., 'r_var_pos': 0.1
    },
    matching_fn_kwargs={'min_iou': 0.25, 'multi_match_min_iou': 0.93}
)

# === ESTADO DE TRACKING ===
first_track = True
track_dict = {}
tracks_to_follow = []
frame_tracks = []
frame_count = 0

# === PROCESAMIENTO DE FRAMES ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección local con YOLOv8
    results = model(frame)[0]
    detections = []

    for result in results.boxes:
        cls_name = model.names[int(result.cls[0])]
        if cls_name != TARGET_CLASS_NAME:
            continue  # Ignora si no es un pato

        x1, y1, x2, y2 = map(int, result.xyxy[0])
        detection_box = np.array([x1, y1, x2, y2])
        detections.append(Detection(box=detection_box))

    # Paso de MOTPY
    _ = multi_tracker.step(detections)
    tracks = multi_tracker.active_tracks()

    if first_track:
        tracks_to_follow = [track.id for track in tracks]
        for track in tracks:
            track_dict[track.id] = []
        first_track = False

    for i, track in enumerate(tracks):
        track_id = track.id
        box = [int(v) for v in track.box]

        # Visualización y almacenamiento
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colors[i].tolist(), 2)
        cv2.putText(frame, f"ID: {track_id}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        if track_id in tracks_to_follow:
            track_dict[track_id].append(box)
            frame_tracks.append(box)

    cv2.imshow("Seguimiento", frame)
    if cv2.waitKey(1) == 27:  # ESC para salir
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# === EXTRAER Y GUARDAR PATO PRINCIPAL ===
max_track_id = max(track_dict, key=lambda k: len(track_dict[k]))
print(f"Pato más visible: Track ID {max_track_id} con {len(track_dict[max_track_id])} detecciones")

cap = cv2.VideoCapture(VIDEO_PATH)
out = cv2.VideoWriter("tracked_duck.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (200, 200))
frame_index = 0

for box in track_dict[max_track_id]:
    ret, frame = cap.read()
    if not ret:
        break

    cropped = frame[box[1]:box[3], box[0]:box[2]]
    cropped_resized = cv2.resize(cropped, (200, 200))
    out.write(cropped_resized)

    cv2.imshow("Pato Principal", cropped_resized)
    if cv2.waitKey(1) == 27:
        break
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()