import json
import os

# ================= CONFIG =================
JSON_PATH = "../annotations/instances_val2017.json"
IMAGES_DIR = "../images/val"
LABELS_DIR = "../labels/val"
CLASS_ID = 0  # person = 0
# =========================================

os.makedirs(LABELS_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    coco = json.load(f)

# Obtener ID de persona
person_id = next(cat["id"] for cat in coco["categories"] if cat["name"] == "person")

# Mapear image_id -> info
images_info = {img["id"]: img for img in coco["images"]}

# Agrupar anotaciones por imagen
annotations_per_image = {}
for ann in coco["annotations"]:
    if ann["category_id"] == person_id:
        annotations_per_image.setdefault(ann["image_id"], []).append(ann)

# Convertir
for image_id, anns in annotations_per_image.items():
    img = images_info[image_id]
    filename = img["file_name"]
    img_path = os.path.join(IMAGES_DIR, filename)

    if not os.path.exists(img_path):
        continue

    w, h = img["width"], img["height"]
    label_path = os.path.join(LABELS_DIR, filename.replace(".jpg", ".txt"))

    with open(label_path, "w") as f:
        for ann in anns:
            if isinstance(ann["segmentation"], list):
                for seg in ann["segmentation"]:
                    if len(seg) < 6:
                        continue
                    norm = []
                    for i in range(0, len(seg), 2):
                        norm.append(seg[i] / w)
                        norm.append(seg[i + 1] / h)

                    line = str(CLASS_ID) + " " + " ".join(f"{x:.6f}" for x in norm)
                    f.write(line + "\n")