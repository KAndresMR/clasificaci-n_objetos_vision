import json
import os
import requests
from tqdm import tqdm
import random

# ================= CONFIG =================
JSON_PATH = "../annotations/instances_val2017.json"
OUTPUT_DIR = "../images/val"
MAX_IMAGES = 50   # cambia si quieres
COCO_BASE_URL = "http://images.cocodataset.org/val2017/"
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar JSON
with open(JSON_PATH, "r") as f:
    coco = json.load(f)

# Obtener ID de la categoría "person"
person_category_id = None
for cat in coco["categories"]:
    if cat["name"] == "person":
        person_category_id = cat["id"]
        break

assert person_category_id is not None, "No se encontró la categoría 'person'"

# Obtener IDs de imágenes que contienen personas
image_ids_with_person = set()
for ann in coco["annotations"]:
    if ann["category_id"] == person_category_id:
        image_ids_with_person.add(ann["image_id"])

image_ids_with_person = list(image_ids_with_person)
random.shuffle(image_ids_with_person)
image_ids_with_person = image_ids_with_person[:MAX_IMAGES]

# Mapear image_id -> file_name
id_to_filename = {
    img["id"]: img["file_name"]
    for img in coco["images"]
    if img["id"] in image_ids_with_person
}

# Descargar imágenes
for image_id, filename in tqdm(id_to_filename.items(), desc="Descargando imágenes"):
    url = COCO_BASE_URL + filename
    out_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(out_path):
        continue

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        print(f"Error al descargar {filename}")