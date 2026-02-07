from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
    
model.predict(
    source=r"C:\Users\ramya\Desktop\BO_analysis\Final_Analysis\bdd100k_images_100k\bdd100k\images\100k\val",
    save=True,
    conf=0.3
)
