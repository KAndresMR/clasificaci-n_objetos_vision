import cv2
import time
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO("../models/best.pt")  # ajusta la ruta si hace falta

cap = cv2.VideoCapture(0)

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    results = model.predict(
        source=frame,
        device="cpu",
        conf=0.4,
        stream=False
    )

    annotated_frame = results[0].plot()

    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv12 Segmentation (CPU)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()