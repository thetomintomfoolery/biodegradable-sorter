from ultralytics import YOLO

model = YOLO("runs/detect/train24/weights/best.pt")

# Predict and save labels in YOLO format
results = model.predict(
    source="new_images/",   # folder with your new images
    save=True,              # save images with bounding boxes
    save_txt=True,          # save labels in YOLO format
    conf=0.5,               # minimum confidence to consider a detection
    imgsz=640
)
