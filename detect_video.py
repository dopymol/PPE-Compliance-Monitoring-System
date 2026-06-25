import cv2
from pathlib import Path
from ultralytics import YOLO

models = YOLO('models/best.pt')

video_path = 'input/videos/sample.mp4'
output_path = 'output/videos/ppe_detected_video.mp4'

if not Path(video_path).exists():
    print("Video not found. Add a video as input/videos/sample.mp4")
    exit()

Path("output/videos").mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    if not success:
        break

    frame_count += 1

    results = models(frame, conf=0.45, iou=0.4)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow("PPE Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video PPE detection completed.")
print(f"Processed frames: {frame_count}")
print(f"Output saved at: {output_path}")