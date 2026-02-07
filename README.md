# BDD100K Object Detection using YOLOv8

This project demonstrates an end-to-end pipeline for object detection using the BDD100K dataset.  
It includes data analysis, dataset conversion, model training, evaluation, and visualization.

---

## 1. Project Structure

bdd-object-detection-analysis/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
├── scripts/
│   ├── bdd_analysis_simple_28.py
│   ├── bdd_to_yolo.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── visualize.py
│
├── bdd.yaml
├── runs/
│   └── detect/train2/weights/best.pt
│
└── README.md

---

## 2. Dataset

Dataset used: BDD100K (Berkeley DeepDrive 100K)  
Website: https://bdd-data.berkeley.edu/

We use only the object detection annotations (bounding boxes).  
Semantic segmentation and lane data are ignored.

---

## 3. Data Analysis

Script:

python scripts/bdd_analysis_simple_28.py

This performs:
- JSON parsing  
- Class distribution analysis  
- Train vs Validation comparison  
- Bounding box anomaly detection  
- Interactive dashboard  
- Visualization of sample images  

For fast execution, analysis is done on first 28 images only.

---

## 4. Dataset Conversion (JSON → YOLO)

Script:

python scripts/bdd_to_yolo.py

This converts:
BDD JSON labels → YOLO format .txt labels

Saved in:

labels/train/  
labels/val/  
labels/test/

---

## 5. Model Choice

We use YOLOv8 (Ultralytics).

Why YOLOv8:
- Single-stage detector (fast)
- Pretrained on COCO
- Industry standard
- Easy training and deployment

---

## 6. Training

Script:

python scripts/train_model.py

Model saved at:

runs/detect/train2/weights/best.pt

---

## 7. Evaluation

Script:

python scripts/evaluate_model.py

Metrics used:
- Precision
- Recall
- mAP@0.5
- F1-score

These metrics are standard for object detection.

---

## 8. Visualization

Script:

python scripts/visualize.py

Shows:
- Ground truth boxes
- Predicted boxes
- Failure cases
- Missed detections

This is qualitative analysis.

---

## 9. Key Observations

From analysis and evaluation:

- Dataset is highly imbalanced (car dominates)
- Small objects have lower recall
- Occluded pedestrians are hardest
- Label quality impacts performance heavily

---

## 10. How to Run Everything (Order)

Run scripts in this order:

1. Data analysis  
python scripts/bdd_analysis_simple_28.py  

2. Convert dataset  
python scripts/bdd_to_yolo.py  

3. Train model  
python scripts/train_model.py  

4. Evaluate  
python scripts/evaluate_model.py  

5. Visualize  
python scripts/visualize.py  

---

## 11. Docker (Optional)

Build:

docker build -t bdd-yolo .

Run:

docker run -it bdd-yolo

---

## 12. Future Improvements

- Train on full dataset
- Data augmentation
- Use other models to improve the detection along with segmentation concepts
- Class rebalancing


---

