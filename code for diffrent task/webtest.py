from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

model =  YOLO("../Yolo-Weights/yolov8n.pt")
while True:
    success, img = cap.read()
    result = model(img, stream=True)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
