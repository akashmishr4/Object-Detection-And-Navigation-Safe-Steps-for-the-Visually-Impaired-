from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
default_rate = engine.getProperty('rate')
engine.setProperty('rate', int(default_rate * 0.5))  # Set slower speech rate

# Debug: Verify TTS initialization
print("Text-to-Speech engine initialized.")

# Set up webcam
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Object class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Define target classes for road-related alert
road_related_classes = {"bicycle", "car", "traffic light", "stop sign"}

# Variables for managing alerts
prev_frame_time = 0
new_frame_time = 0
last_alert_time = 0
# alert_cooldown = 5  # seconds

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    current_detected = set()  # Track objects detected in the current frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            current_detected.add(class_name)

            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Check for road-related detections and play an alert
    if road_related_classes.intersection(current_detected):
        if time.time() - last_alert_time > alert_cooldown:  # Check cooldown
            last_alert_time = time.time()
            print("Alert: We are near a road.")  # Debug print
            engine.say("We are near a road.")
            engine.runAndWait()  # Ensure this processes correctly
            print("TTS executed.")  # Debug print

    # Update the frame rate
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
