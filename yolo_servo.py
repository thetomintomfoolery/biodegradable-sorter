import serial
import time
from ultralytics import YOLO

# ---------------- Arduino Setup ----------------
arduino = serial.Serial('COM4', 9600, timeout=1)  # Arduino COM port
time.sleep(2)  # Wait for Ardui0no to initialize

# ---------------- YOLO Setup ----------------
model = YOLO("runs/detect/train32/weights/best.pt")  # Your trained model

# ---------------- Function to send servo command ----------------
def send_class(label):
    """Send classification result to Arduino"""
    if label == "biodegradable":
        arduino.write(b'B')
    elif label == "non-biodegradable":
        arduino.write(b'N')

# ---------------- Webcam detection loop ----------------
results_stream = model.predict(
    source=1,    # Webcam
    stream=True, # Stream frames
    conf=0.4,    # Confidence threshold
    device=0     # GPU/CPU
)

for result in results_stream:
    # result.boxes.cls gives class index, result.boxes.conf gives confidence
    # If your YOLO dataset has 0 = biodegradable, 1 = non-biodegradable:
    if len(result.boxes) > 0:
        for cls in result.boxes.cls:
            cls_index = int(cls)
            if cls_index == 0:
                send_class("biodegradable")
                print("Sent: Biodegradable")
            elif cls_index == 1:
                send_class("non-biodegradable")
                print("Sent: Non-biodegradable")
