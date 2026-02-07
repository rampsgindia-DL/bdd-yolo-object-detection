#This file used to convert BDD100K JSON annotations to YOLO format
import json
import os
import cv2
from tqdm import tqdm

IMAGES_ROOT = r"C:\Users\ramya\Desktop\BO_analysis\Final_Analysis\bdd100k_images_100k\bdd100k\images\100k"
LABELS_JSON = r"C:\Users\ramya\Desktop\BO_analysis\Final_Analysis\bdd100k_labels_release\bdd100k\labels"

YOLO_ROOT = r"C:\Users\ramya\Desktop\BO_analysis\Final_Analysis\scripts\bdd100k_yolo"
os.makedirs(YOLO_ROOT, exist_ok=True)

MAX_IMAGES = 28000

for split in ["train", "val"]:
    os.makedirs(f"{YOLO_ROOT}/images/{split}", exist_ok=True)
    os.makedirs(f"{YOLO_ROOT}/labels/{split}", exist_ok=True)

CLASSES = [
    "person", "car", "bus", "truck", "bike",
    "motor", "traffic light", "traffic sign",
    "train", "rider"
]
cls_map = {c: i for i, c in enumerate(CLASSES)}

def convert(split):
    json_file = os.path.join(LABELS_JSON, f"bdd100k_labels_images_{split}.json")
    with open(json_file) as f:
        data = json.load(f)

    for img in tqdm(data[:MAX_IMAGES], desc=split):
        name = img["name"]
        src_img = os.path.join(IMAGES_ROOT, split, name)
        dst_img = os.path.join(YOLO_ROOT, "images", split, name)

        if not os.path.exists(src_img):
            continue

        os.system(f'copy "{src_img}" "{dst_img}"')

        image = cv2.imread(src_img)
        h, w = image.shape[:2]

        yolo_lines = []
        for label in img.get("labels", []):
            if "box2d" not in label:
                continue
            cat = label["category"]
            if cat not in cls_map:
                continue

            b = label["box2d"]
            xc = ((b["x1"] + b["x2"]) / 2) / w
            yc = ((b["y1"] + b["y2"]) / 2) / h
            bw = (b["x2"] - b["x1"]) / w
            bh = (b["y2"] - b["y1"]) / h

            yolo_lines.append(
                f"{cls_map[cat]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

        label_path = os.path.join(YOLO_ROOT, "labels", split, name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

convert("train")
convert("val")
print("DONE")

