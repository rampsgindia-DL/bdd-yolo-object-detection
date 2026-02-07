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
