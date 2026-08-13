from ultralytics import YOLO
from pathlib import Path
from collections import Counter

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

OLD_MODEL = "runs/yolo26/full_training/weights/best.pt"

NEW_MODEL = (
    "runs/detect/runs/yolo26/"
    "corrected_20_retrain-3/weights/best.pt"
)

TEST_IMAGES = Path("Dataset/test/images")
TEST_LABELS = Path("Dataset/test/labels")

# Your class mapping
CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC"
}

CONF = 0.5


# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------

print("Loading models...")

old_model = YOLO(OLD_MODEL)
new_model = YOLO(NEW_MODEL)


# -------------------------------------------------------
# COUNT GROUND-TRUTH LABELS
# -------------------------------------------------------

def get_ground_truth(label_path):
    counts = Counter()

    if not label_path.exists():
        return counts

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                counts[class_id] += 1

    return counts


# -------------------------------------------------------
# RUN PREDICTION
# -------------------------------------------------------

def get_predictions(model, image_path):
    counts = Counter()

    results = model.predict(
        source=str(image_path),
        conf=CONF,
        verbose=False
    )

    for result in results:
        if result.boxes is not None:
            for class_id in result.boxes.cls.tolist():
                counts[int(class_id)] += 1

    return counts


# -------------------------------------------------------
# TOTAL COUNTS
# -------------------------------------------------------

gt_total = Counter()
old_total = Counter()
new_total = Counter()

image_files = sorted(
    p for p in TEST_IMAGES.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

print(f"\nTesting on {len(image_files)} images")
print(f"Confidence threshold: {CONF}\n")


for i, image_path in enumerate(image_files, start=1):

    label_path = TEST_LABELS / f"{image_path.stem}.txt"

    gt = get_ground_truth(label_path)
    old = get_predictions(old_model, image_path)
    new = get_predictions(new_model, image_path)

    gt_total.update(gt)
    old_total.update(old)
    new_total.update(new)

    print(
        f"[{i:3}/{len(image_files)}] "
        f"{image_path.name}"
    )


# -------------------------------------------------------
# FINAL RESULTS
# -------------------------------------------------------

print("\n")
print("=" * 75)
print("OLD MODEL vs CORRECTED-DATASET RETRAINED MODEL")
print("=" * 75)

print(
    f"{'Class':<12}"
    f"{'Actual':>10}"
    f"{'Old':>10}"
    f"{'Old Diff':>12}"
    f"{'New':>10}"
    f"{'New Diff':>12}"
)

print("-" * 75)

for class_id in [0, 1, 2]:

    actual = gt_total[class_id]
    old_pred = old_total[class_id]
    new_pred = new_total[class_id]

    old_diff = old_pred - actual
    new_diff = new_pred - actual

    print(
        f"{CLASS_NAMES[class_id]:<12}"
        f"{actual:>10}"
        f"{old_pred:>10}"
        f"{old_diff:>+12}"
        f"{new_pred:>10}"
        f"{new_diff:>+12}"
    )


print("=" * 75)


# -------------------------------------------------------
# ABSOLUTE COUNT ERROR
# -------------------------------------------------------

print("\nABSOLUTE COUNT ERROR")
print("-" * 45)

for class_id in [0, 1, 2]:

    actual = gt_total[class_id]

    old_error = abs(old_total[class_id] - actual)
    new_error = abs(new_total[class_id] - actual)

    if old_error == 0:
        improvement = 0
    else:
        improvement = (
                              (old_error - new_error) / old_error
                      ) * 100

    print(
        f"{CLASS_NAMES[class_id]:<12}"
        f" Old={old_error:<6}"
        f" New={new_error:<6}"
        f" Improvement={improvement:>7.2f}%"
    )

print("\nDone.")