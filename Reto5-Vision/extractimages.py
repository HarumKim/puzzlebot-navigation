import cv2

cap = cv2.VideoCapture("tracked_duck.avi")
count = 0

# Definir la nueva resolución que deseas (por ejemplo, duplicar el tamaño)
new_width = 1920  # ancho de la imagen deseado
new_height = 1080  # altura de la imagen deseada

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if count % 10 == 0:
        # Redimensionar el frame a la nueva resolución
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # Guardar el frame redimensionado
        cv2.imwrite(f"FRAMES/frame_{count:04d}.jpg", resized_frame)
    
    count += 1

cap.release()
