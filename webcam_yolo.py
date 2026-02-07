import cv2
import time
import serial
from ultralytics import YOLO
from collections import deque


# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "runs/classify/train8/weights/best.pt"

CONF_THRESH = 0.85
SMOOTHING_FRAMES = 5
STABLE_COUNT = 5

CAMERA_INDEX = 1

# Arduino
ARDUINO_PORT = "COM5"   # CHANGE THIS
BAUD_RATE = 9600


# Material → Category mapping
BIO = {"paper", "cardboard"}
NON_BIO = {"plastic", "metal", "sachet", "styrofoam"}
IGNORE = {"background"}


# =========================
# MAIN FUNCTION
# =========================
def main():
    model = YOLO(MODEL_PATH)

    print("\n✅ Model classes:")
    for i, name in model.names.items():
        print(f"  {i}: {name}")
    print()

    # Arduino connection
    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Arduino connected on {ARDUINO_PORT}")
    except Exception as e:
        print("❌ Arduino error:", e)
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Webcam not found")
        return

    history = deque(maxlen=SMOOTHING_FRAMES)
    object_locked = False   # 🔐 VERY IMPORTANT

    print("🎥 Smart Bin AI running (press Q to quit)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        cls_id = results.probs.top1
        conf = results.probs.top1conf.item()
        cls_name = results.names[cls_id]

        history.append(cls_name)
        stable = history.count(cls_name) >= STABLE_COUNT

        # =========================
        # DECISION LOGIC
        # =========================
        if conf < CONF_THRESH or not stable or cls_name in IGNORE:
            decision = "NO OBJECT"
            color = (180, 180, 180)

            # 🔓 Unlock when object disappears
            object_locked = False

        elif cls_name in BIO:
            decision = "BIODEGRADABLE"
            color = (0, 200, 0)

        elif cls_name in NON_BIO:
            decision = "NON-BIODEGRADABLE"
            color = (0, 0, 255)

        else:
            decision = "UNKNOWN"
            color = (255, 255, 255)

        # =========================
        # SERVO CONTROL (ONE-SHOT)
        # =========================
        if not object_locked:
            if decision == "BIODEGRADABLE":
                arduino.write(b'B')
                print("➡️ Servo moved: BIO")
                object_locked = True

            elif decision == "NON-BIODEGRADABLE":
                arduino.write(b'N')
                print("➡️ Servo moved: NON-BIO")
                object_locked = True

        # =========================
        # DISPLAY
        # =========================
        label = f"{cls_name.upper()} ({conf:.2f})"

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        cv2.putText(frame, decision, (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    color, 3)

        cv2.imshow("Smart Bin AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    arduino.close()
    cv2.destroyAllWindows()


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
