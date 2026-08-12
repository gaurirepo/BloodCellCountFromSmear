from ultralytics import YOLO
from pathlib import Path

print("=" * 60)
print("BLOOD CELL MODEL ACCURACY TEST")
print("=" * 60)

# --------------------------------------------------
# PROJECT LOCATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = PROJECT_DIR / "best.pt"

# Change this later to the YOLO26 model you want to test
# For now, if best.pt is your current trained model,
# this will use it.

# --------------------------------------------------
# DATASET
# --------------------------------------------------

DATA_YAML = PROJECT_DIR / "data.yaml"

TEST_IMAGES = PROJECT_DIR / "Dataset/test/images"

TEST_LABELS = PROJECT_DIR / "Dataset/test/labels"

# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

print("\nChecking files...")

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not DATA_YAML.exists():

    raise FileNotFoundError(
        f"\ndata.yaml not found:\n{DATA_YAML}"
    )

if not TEST_IMAGES.exists():

    raise FileNotFoundError(
        f"\nTest image folder not found:\n{TEST_IMAGES}"
    )

if not TEST_LABELS.exists():

    raise FileNotFoundError(
        f"\nTest label folder not found:\n{TEST_LABELS}"
    )

print("Model       : OK")
print("data.yaml   : OK")
print("Test images : OK")
print("Test labels : OK")

# --------------------------------------------------
# COUNT TEST IMAGES
# --------------------------------------------------

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

images = [
    f for f in TEST_IMAGES.iterdir()
    if f.suffix.lower() in image_extensions
]

print("\nTest images found:", len(images))

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")

print("\nModel classes:")

print(model.names)

# --------------------------------------------------
# YOLO STANDARD EVALUATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("RUNNING YOLO TEST EVALUATION")
print("=" * 60)

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    conf=0.001,
    iou=0.7,
    plots=True
)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map5095 = metrics.box.map

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(f"\nPrecision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"mAP50     : {map50 * 100:.2f}%")
print(f"mAP50-95  : {map5095 * 100:.2f}%")

# --------------------------------------------------
# F1
# --------------------------------------------------

if precision + recall > 0:

    f1 = (
            2 * precision * recall
            / (precision + recall)
    )

else:

    f1 = 0

print(f"F1        : {f1 * 100:.2f}%")

# --------------------------------------------------
# CLASS RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASS RESULTS")
print("=" * 60)

for i, name in model.names.items():

    p = metrics.box.p[i]

    r = metrics.box.r[i]

    ap50 = metrics.box.ap50[i]

    ap = metrics.box.ap[i]

    if p + r > 0:

        class_f1 = (
                2 * p * r
                / (p + r)
        )

    else:

        class_f1 = 0

    print(f"\n{name}")

    print(f"  Precision : {p * 100:.2f}%")
    print(f"  Recall    : {r * 100:.2f}%")
    print(f"  F1        : {class_f1 * 100:.2f}%")
    print(f"  mAP50     : {ap50 * 100:.2f}%")
    print(f"  mAP50-95  : {ap * 100:.2f}%")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)