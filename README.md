# bdd-yolo-object-detection
A complete computer vision project implementing object detection on the BDD100K dataset using YOLOv8. Includes dataset analysis, JSON-to-YOLO conversion, model training, quantitative evaluation, and qualitative visualization.

**Detailed Project Description**

This project implements a complete end-to-end object detection system using the BDD100K (Berkeley DeepDrive 100K) dataset and the YOLOv8 deep learning model.

The goal of the project is to design a reproducible machine learning pipeline that covers the full lifecycle of a real-world computer vision system, including:

 * Dataset exploration and understanding
 * Data preprocessing and format conversion
 * Model selection and training
 * Quantitative evaluation using standard metrics
 * Qualitative visualization of model performance
 * Documentation and reproducibility

**Problem Statement**

Autonomous driving systems rely heavily on accurate perception of objects such as cars, pedestrians, cyclists, traffic signs, and traffic lights. The BDD100K dataset provides large-scale annotated driving images to simulate this real-world scenario.

This project focuses on solving the object detection problem, where the task is to:

  * Detect and localize objects in road images by predicting bounding boxes and class labels.

**Model Selection**

The chosen model is **YOLOv8 (You Only Look Once v8) **from Ultralytics.

YOLOv8 is selected because:

 *It is a single-stage detector, enabling real-time performance
 *It provides strong accuracy with low latency
 *It comes with pretrained weights on COCO
 *It is widely used in industry and research
 *It supports easy training and deployment
 *The project uses transfer learning by fine-tuning pretrained YOLOv8 weights on the BDD100K dataset.

 **Training Pipeline**

The training pipeline includes:

*Custom dataset loader using YOLO format
*Pretrained backbone initialization
*One-epoch training (for demonstration)
*Automatic checkpoint saving
*GPU/CPU compatibility
The trained model is stored under:

 runs/detect/train2/weights/best.pt

**Evaluation**
The model is evaluated on the validation dataset using standard object detection metrics:

 *Precision – correctness of predicted detections
 *Recall – ability to find all relevant objects
 *mAP@0.5 – mean average precision at IoU threshold 0.5
 *F1-score – harmonic mean of precision and recall

These metrics are chosen because they are widely accepted benchmarks in object detection research and industry.

**Key Findings
**
From both quantitative and qualitative analysis:

 **Severe class imbalance** exists in BDD100K, with the car class accounting for the majority of annotations. This imbalance leads to biased learning and reduced performance on minority classes such as traffic signs and riders.

 **Object scale strongly affects detection performance**. The model achieves high precision for large objects (cars, buses, trucks) but struggles significantly with small-scale objects like traffic lights and distant pedestrians.

 **Occlusion is a dominant failure factor.** Pedestrians partially hidden by vehicles or infrastructure are frequently missed, indicating limitations in the model’s ability to reason under visual obstruction.

 **High false positives occur in visually ambiguous regions**, such as reflections, shadows, and overlapping objects, highlighting the sensitivity of single-stage detectors to complex visual patterns.
