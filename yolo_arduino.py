from ultralytics import YOLO
import cv2
import serial
import time

model = YOLO("runs/detect/train38/weights/best.pt")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

arduino = serial.Serial("COM6", 9600, timeout=1)
time.sleep(2)

last_sent = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6)

    detected_label = None

    for r in results:
        for box in r.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label + confidence
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detected_label = label
            break  # take only first detection

    # Send to Arduino ONLY if changed
    if detected_label and detected_label != last_sent:
        if detected_label == "biodegradable":
            arduino.write(b'B\n')
            last_sent = "biodegradable"
        elif detected_label == "non-biodegradable":
            arduino.write(b'N\n')
            last_sent = "non-biodegradable"

    cv2.imshow("YOLOv8", frame)
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
