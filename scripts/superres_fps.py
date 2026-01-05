import cv2
import time
import torch
import numpy as np

from realesrgan import RealESRGAN

# -----------------------------
# Configuración
# -----------------------------
VIDEO_INPUT = "video_input.mp4"
VIDEO_OUTPUT = "video_superres_output.mp4"
SCALE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Ejecutando en: {DEVICE}")

# -----------------------------
# Cargar modelo Real-ESRGAN
# -----------------------------
model = RealESRGAN(DEVICE, scale=SCALE)
model.load_weights("RealESRGAN_x4plus.pth", download=True)

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
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # OpenCV -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Super resolución
    sr_frame = model.predict(frame_rgb)

    # RGB -> BGR
    sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)

    # FPS
    end = time.time()
    fps = 1 / (end - start)

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
    cv2.imshow("Super Resolution - RealESRGAN", sr_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# Liberar recursos
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("[INFO] Procesamiento finalizado")