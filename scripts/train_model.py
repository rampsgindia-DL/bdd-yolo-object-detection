"""
Minimal YOLOv8 training pipeline for BDD100K.
Trains for 1 epoch on a subset of data.
"""

from ultralytics import YOLO


def train_one_epoch():
    """
    Load pretrained YOLOv8 model and train for 1 epoch.
    """
    model = YOLO("yolov8n.pt")  # lightweight model

    model.train(
        data=r"C:\Users\ramya\Desktop\BO_analysis\Final_Analysis\scripts\bdd.yaml",
        epochs=1,
        imgsz=256,
        batch=4,
        device="cpu",
        workers=2
    )


if __name__ == "__main__":
    train_one_epoch()
