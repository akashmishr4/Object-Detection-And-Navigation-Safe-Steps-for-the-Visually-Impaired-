from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set the speech rate to 80% of the default rate
default_rate = engine.getProperty('rate')
engine.setProperty('rate', int(default_rate * 0.5))

# Set up webcam
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

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

prev_frame_time = 0
new_frame_time = 0
detected_objects = {}  # To track detected objects and their positions

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    height, width, _ = img.shape  # Get frame dimensions
    results = model(img, stream=True)
    current_detected = {}  # Track objects detected in the current frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            x_center, y_center = x1 + w // 2, y1 + h // 2  # Calculate the center of the box

            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Determine the position
            position = ""
            if x_center < width // 3:
                position += "left "
            elif x_center > 2 * width // 3:
                position += "right "
            else:
                position += "center "

            if y_center < height // 3:
                position += "top"
            elif y_center > 2 * height // 3:
                position += "bottom"
            else:
                position += "middle"

            current_detected[class_name] = position

            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Check for new detections and announce them with positions
    for obj, position in current_detected.items():
        if obj not in detected_objects or detected_objects[obj] != position:
            detected_objects[obj] = position
            engine.say(f"I see a {obj} at the {position}.")
            engine.runAndWait()

    # Update the frame rate
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
