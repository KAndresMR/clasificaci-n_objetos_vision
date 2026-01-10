import cv2
import time
import numpy as np

# -----------------------------
# Configuración
# -----------------------------
VIDEO_INPUT = "video_input.mp4"
VIDEO_OUTPUT = "video_superres_output.mp4"

MODEL_PATH = "EDSR_x4.pb"   # Modelo preentrenado
SCALE = 4

# -----------------------------
# Inicializar Super-Resolution
# -----------------------------
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_PATH)
sr.setModel("edsr", SCALE)

print("[INFO] Modelo EDSR x4 cargado")

# -----------------------------
# Cargar video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_INPUT)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

out_width = width * SCALE
out_height = height * SCALE

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    fps_input,
    (out_width, out_height)
)

# -----------------------------
# Procesamiento frame a frame
# -----------------------------
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # Super resolución
    sr_frame = sr.upsample(frame)

    end = time.time()
    fps = 1 / (end - start)
    fps_list.append(fps)

    # Dibujar FPS
    cv2.putText(
        sr_frame,
        f"FPS: {fps:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    out.write(sr_frame)
    cv2.imshow("Super Resolution - OpenCV EDSR", sr_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# Liberar recursos
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

avg_fps = sum(fps_list) / len(fps_list)
print(f"[INFO] FPS promedio: {avg_fps:.2f}")
print("[INFO] Procesamiento finalizado")
