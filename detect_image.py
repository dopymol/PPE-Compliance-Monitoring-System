from ultralytics import YOLO
from pathlib import Path
import pandas as pd

from ppe_matcher import assign_ppe_to_workers


# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "models/best.pt"
IMAGE_PATH = "input/images/test_image1.jpg"
REPORT_PATH = "output/reports/worker_ppe_report.csv"

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.40


# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO(MODEL_PATH)


# -------------------------------
# Check input image
# -------------------------------
if not Path(IMAGE_PATH).exists():
    print(f"Image not found. Add an image as {IMAGE_PATH}")
    exit()


# -------------------------------
# Run PPE detection
# -------------------------------
results = model(
    IMAGE_PATH,
    conf=CONFIDENCE_THRESHOLD,
    iou=IOU_THRESHOLD,
    save=True
)

persons = []
ppe_objects = []

print("\nRaw YOLO Detections:")

for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = results[0].names[class_id].lower()
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    detection = {
        "class_name": class_name,
        "confidence": round(confidence, 2),
        "box": [x1, y1, x2, y2]
    }

    print(
        class_name,
        round(confidence, 2),
        [round(x1), round(y1), round(x2), round(y2)]
    )

    if class_name == "person":
        persons.append(detection)
    else:
        ppe_objects.append(detection)


# -------------------------------
# Handle no workers found
# -------------------------------
if len(persons) == 0:
    print("\nNo workers/persons detected in the image.")
    exit()


# -------------------------------
# Assign PPE objects to workers
# -------------------------------
worker_ppe = assign_ppe_to_workers(persons, ppe_objects)

print("\nWorker-wise PPE Compliance Report")

report_data = []


# -------------------------------
# Worker-wise compliance checking
# -------------------------------
for idx, person in enumerate(persons, start=1):

    ppe = worker_ppe[idx - 1]

    has_helmet = ppe["helmet"]
    has_vest = ppe["vest"]
    has_gloves = ppe["gloves"]
    has_goggles = ppe["goggles"]
    has_boots = ppe["boots"]

    has_no_helmet = ppe["no_helmet"]
    has_no_gloves = ppe["no_gloves"]
    has_no_goggle = ppe["no_goggle"]

    violations = []

    # Mandatory PPE rules
    # Helmet is marked missing only when no_helmet is detected.
    # This avoids false violation when helmet is simply missed by the model.
    if has_no_helmet:
        violations.append("Helmet missing")

    if not has_vest:
        violations.append("Vest missing")

    # Optional violation classes
    if has_no_gloves:
        violations.append("Gloves missing")

    if has_no_goggle:
        violations.append("Goggles missing")

    status = "Violation" if violations else "Compliant"

    print(f"\nWorker {idx}")
    print(f"Helmet: {'Yes' if has_helmet else 'Not confidently detected'}")
    print(f"Vest: {'Yes' if has_vest else 'No'}")
    print(f"Gloves: {'Yes' if has_gloves else 'Not confidently detected'}")
    print(f"Goggles: {'Yes' if has_goggles else 'Not confidently detected'}")
    print(f"Boots/Shoes: {'Yes' if has_boots else 'Not confidently detected'}")
    print(f"Status: {status}")

    if violations:
        print("Violations:")
        for violation in violations:
            print(f"- {violation}")

    report_data.append({
        "Worker_ID": idx,
        "Helmet": "Yes" if has_helmet else "Not confidently detected",
        "Vest": "Yes" if has_vest else "No",
        "Gloves": "Yes" if has_gloves else "Not confidently detected",
        "Goggles": "Yes" if has_goggles else "Not confidently detected",
        "Boots/Shoes": "Yes" if has_boots else "Not confidently detected",
        "Status": status,
        "Violations": "; ".join(violations) if violations else "None"
    })


# -------------------------------
# Save CSV report
# -------------------------------
Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(report_data)
df.to_csv(REPORT_PATH, index=False)


print("\nDetection completed.")
print("Annotated image saved inside runs/detect/predict/")
print(f"Worker PPE report saved at {REPORT_PATH}")