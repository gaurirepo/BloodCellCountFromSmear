from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
        PROJECT_DIR
        / "runs/yolo26/full_training/weights/best.pt"
)

IMAGE_PATH = (
        PROJECT_DIR
        / "Dataset/test/images/"
        / "BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg"
)

LABEL_PATH = (
        PROJECT_DIR
        / "Dataset/test/labels/"
        / "BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.txt"
)

# IoU threshold for deciding whether a prediction
# matches a ground-truth box.
IOU_THRESHOLD = 0.50

# Use this confidence for inspection.
CONFIDENCE = 0.25

CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC",
}


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
            intersection_width *
            intersection_height
    )

    area1 = (
            (box1[2] - box1[0]) *
            (box1[3] - box1[1])
    )

    area2 = (
            (box2[2] - box2[0]) *
            (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SINGLE IMAGE GROUND-TRUTH vs PREDICTION ANALYSIS")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nImage:")
print(IMAGE_PATH)

print("\nLabel:")
print(LABEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found:\n{IMAGE_PATH}")

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"Label not found:\n{LABEL_PATH}")


model = YOLO(str(MODEL_PATH))

print("\nModel classes:")
print(model.names)


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError("Could not read image.")

height, width = image.shape[:2]

print("\nImage dimensions:")
print(f"{width} x {height}")


# ============================================================
# READ GROUND TRUTH
# ============================================================

ground_truth = []

with open(LABEL_PATH, "r") as f:

    for line in f:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        ground_truth.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "box": [x1, y1, x2, y2],
        })


# ============================================================
# GROUND TRUTH COUNTS
# ============================================================

gt_counts = Counter(
    item["class_name"]
    for item in ground_truth
)

print("\n" + "=" * 70)
print("GROUND TRUTH")
print("=" * 70)

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_name:12s}: "
        f"{gt_counts[class_name]}"
    )


# ============================================================
# RUN MODEL
# ============================================================

print("\n" + "=" * 70)
print(f"MODEL PREDICTIONS @ CONFIDENCE {CONFIDENCE}")
print("=" * 70)

results = model.predict(
    source=str(IMAGE_PATH),
    conf=CONFIDENCE,
    imgsz=640,
    device="cpu",
    verbose=False,
)

result = results[0]

predictions = []

for box in result.boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    xyxy = box.xyxy[0].tolist()

    predictions.append({
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": confidence,
        "box": xyxy,
    })


pred_counts = Counter(
    item["class_name"]
    for item in predictions
)

print("\nPredicted counts:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_name:12s}: "
        f"{pred_counts[class_name]}"
    )


# ============================================================
# MATCH PREDICTIONS TO GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("MATCHING PREDICTIONS TO GROUND TRUTH")
print("=" * 70)

matched_gt = set()

correct = Counter()
wrong_class = Counter()
false_positive = Counter()
missed = Counter()

for pred_index, prediction in enumerate(predictions):

    best_iou = 0
    best_gt_index = None

    for gt_index, gt in enumerate(ground_truth):

        if gt_index in matched_gt:
            continue

        iou = calculate_iou(
            prediction["box"],
            gt["box"]
        )

        if iou > best_iou:
            best_iou = iou
            best_gt_index = gt_index

    if (
            best_gt_index is not None
            and best_iou >= IOU_THRESHOLD
    ):

        gt = ground_truth[best_gt_index]

        matched_gt.add(best_gt_index)

        if (
                prediction["class_id"]
                ==
                gt["class_id"]
        ):

            correct[prediction["class_name"]] += 1

            print(
                f"CORRECT   "
                f"{prediction['class_name']:10s} "
                f"confidence={prediction['confidence']:.3f} "
                f"IoU={best_iou:.3f}"
            )

        else:

            wrong_class[
                gt["class_name"]
            ] += 1

            print(
                f"WRONG     "
                f"Predicted={prediction['class_name']:10s} "
                f"Actual={gt['class_name']:10s} "
                f"confidence={prediction['confidence']:.3f} "
                f"IoU={best_iou:.3f}"
            )

    else:

        false_positive[
            prediction["class_name"]
        ] += 1

        print(
            f"FALSE POS "
            f"{prediction['class_name']:10s} "
            f"confidence={prediction['confidence']:.3f}"
        )


# ============================================================
# MISSED GROUND TRUTH
# ============================================================

for gt_index, gt in enumerate(ground_truth):

    if gt_index not in matched_gt:

        missed[
            gt["class_name"]
        ] += 1

        print(
            f"MISSED    "
            f"{gt['class_name']}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)

print("\nCorrect detections:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_name:12s}: "
        f"{correct[class_name]}"
    )


print("\nWrong classifications:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_name:12s}: "
        f"{wrong_class[class_name]}"
    )


print("\nFalse positives:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_name:12s}: "
        f"{false_positive[class_name]}"
    )


print("\nMissed cells:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_name:12s}: "
        f"{missed[class_name]}"
    )


print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

total_gt = len(ground_truth)
total_correct = sum(correct.values())

if total_gt > 0:

    detection_accuracy = (
                                 total_correct / total_gt
                         ) * 100

    print(
        f"\nMatched ground-truth cells: "
        f"{total_correct}/{total_gt}"
    )

    print(
        f"Detection accuracy @ IoU "
        f"{IOU_THRESHOLD}: "
        f"{detection_accuracy:.2f}%"
    )

print("\nDone.")