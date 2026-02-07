"""
BDD100K Object Detection - Simple Data Analysis

This script performs:
1. JSON parsing
2. Class distribution
3. Train vs Val comparison
4. Anomaly detection (tiny/huge boxes)
5. Visualization dashboard
6. Interesting sample visualization
"""

import json
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# -----------------------------
# PATHS (CHANGE ONLY THESE)
# -----------------------------

IMAGES_ROOT = r"C:\Users\ramya\Desktop\BO_analysis\Final_Analysis\bdd100k_images_100k\bdd100k\images\100k"
LABELS_ROOT = r"C:\Users\ramya\Desktop\BO_analysis\Final_Analysis\bdd100k_labels_release\bdd100k\labels"

TRAIN_IMAGES = os.path.join(IMAGES_ROOT, "train")
VAL_IMAGES = os.path.join(IMAGES_ROOT, "val")

TRAIN_JSON = os.path.join(LABELS_ROOT, "bdd100k_labels_images_train.json")
VAL_JSON = os.path.join(LABELS_ROOT, "bdd100k_labels_images_val.json")


# -----------------------------
# 1. JSON PARSER
# -----------------------------

def load_bdd(json_path: str) -> pd.DataFrame:
    """Load BDD JSON and return DataFrame with boxes."""
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for img in data:
        for label in img.get("labels", []):
            if "box2d" not in label:
                continue
            box = label["box2d"]
            rows.append({
                "image": img["name"],
                "category": label["category"],
                "x1": box["x1"],
                "y1": box["y1"],
                "x2": box["x2"],
                "y2": box["y2"],
            })

    return pd.DataFrame(rows)


train_df = load_bdd(TRAIN_JSON)
val_df = load_bdd(VAL_JSON)

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))


# -----------------------------
# 2. CLASS DISTRIBUTION
# -----------------------------

print("\nTRAIN CLASS DISTRIBUTION")
print(train_df["category"].value_counts())


# -----------------------------
# 3. TRAIN vs VAL SPLIT
# -----------------------------

split_compare = (
    train_df["category"].value_counts()
    .to_frame("train")
    .join(val_df["category"].value_counts().to_frame("val"))
    .fillna(0)
)

print("\nTRAIN vs VAL")
print(split_compare)


# -----------------------------
# 4. ANOMALY ANALYSIS
# -----------------------------

train_df["area"] = (train_df.x2 - train_df.x1) * (train_df.y2 - train_df.y1)

tiny = train_df[train_df["area"] < 300]
huge = train_df[train_df["area"] > 250000]

print("\nTINY OBJECTS:", len(tiny))
print("HUGE OBJECTS:", len(huge))


# -----------------------------
# 5. DASHBOARD
# -----------------------------

counts = train_df["category"].value_counts().reset_index()
counts.columns = ["class", "count"]

fig = px.bar(
    counts,
    x="class",
    y="count",
    title="BDD100K Class Distribution"
)
fig.show()


# -----------------------------
# 6. VISUALIZE SAMPLE
# -----------------------------

def show_sample(image_name: str, df: pd.DataFrame):
    """Show image with bounding boxes."""
    path = os.path.join(TRAIN_IMAGES, image_name)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for _, row in df.iterrows():
        cv2.rectangle(
            img,
            (int(row.x1), int(row.y1)),
            (int(row.x2), int(row.y2)),
            (255, 0, 0), 2
        )

    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Show rare class example
rare_class = train_df[train_df["category"] == "train"]
show_sample(rare_class.iloc[0]["image"], rare_class)
