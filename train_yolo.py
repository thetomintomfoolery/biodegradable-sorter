from ultralytics import YOLO

def main():
    model = YOLO("yolov8m-cls.pt")

    model.train(
        data="dataset",
        epochs=120,
        imgsz=224,
        batch=32,
        workers=0   # IMPORTANT for Windows stability
    )

if __name__ == "__main__":
    main()



