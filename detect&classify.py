from ultralytics import YOLO 

model = YOLO("runs/detect/train43/weights/best.pt")
results = model.predict(
    source=1,
    show=True,
    conf=0.4,
)




