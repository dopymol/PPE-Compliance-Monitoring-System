import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")

cap = cv2.VideoCapture(0)  # 0 means default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Real-time PPE detection started. Press 'q' to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    results = model(frame, conf=0.25, iou=0.4)

    annotated_frame = results[0].plot()

    cv2.imshow("Real-Time PPE Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Real-time detection stopped.")