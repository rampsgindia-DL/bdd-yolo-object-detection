"""
Evaluate trained model.
"""

from ultralytics import YOLO


def evaluate():
    model = YOLO("runs/detect/train/weights/best.pt")
    metrics = model.val()

    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    evaluate()
