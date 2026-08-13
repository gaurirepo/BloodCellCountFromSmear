from ultralytics import YOLO
from pathlib import Path

# ============================================================
# YOLO26 FULL TRAINED MODEL - BLOOD CELL PREDICTION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
        PROJECT_DIR
        / "runs"
        / "yolo26"
        / "full_training"
        / "weights"
        / "best.pt"
)

TEST_IMAGES = (
        PROJECT_DIR
        / "DataSet"
        / "test"
        / "images"
)

OUTPUT_DIR = (
        PROJECT_DIR
        / "runs"
        / "yolo26_predictions"
        / "full_training"
)

# Use lower confidence initially so we don't hide WBC detections
CONFIDENCE = 0.25

# ============================================================
# CORRECT DATASET CLASS MAPPING
# ============================================================

EXPECTED_CLASSES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC"
}

print("=" * 60)
print("YOLO26 FULL TRAINED MODEL - BLOOD CELL PREDICTION")
print("=" * 60)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Test images:")
print(TEST_IMAGES)

# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nERROR: best.pt not found!\n{MODEL_PATH}"
    )

if not TEST_IMAGES.exists():
    raise FileNotFoundError(
        f"\nERROR: Test image folder not found!\n{TEST_IMAGES}"
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")

print()
print("Model classes:")
print(model.names)

# ============================================================
# VERIFY CLASS MAPPING
# ============================================================

print()
print("Checking class mapping...")

model_names = {
    int(k): str(v)
    for k, v in model.names.items()
}

print("Expected:")
print(EXPECTED_CLASSES)

print("Model:")
print(model_names)

if model_names != EXPECTED_CLASSES:

    print()
    print("=" * 60)
    print("ERROR: CLASS MAPPING DOES NOT MATCH!")
    print("=" * 60)

    print()
    print("Expected:")
    print(EXPECTED_CLASSES)

    print()
    print("Found in model:")
    print(model_names)

    print()
    print("DO NOT use this model for final predictions.")
    print("Train a fresh model using the corrected data.yaml.")

    raise ValueError("Incorrect class mapping in model.")

print()
print("Class mapping verified successfully!")

# ============================================================
# FIND TEST IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

images = [
    f
    for f in TEST_IMAGES.iterdir()
    if f.is_file()
       and f.suffix.lower() in image_extensions
]

images.sort()

print()
print("Number of test images:", len(images))

if len(images) == 0:
    raise ValueError("No test images found.")

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print()
print("Output directory:")
print(OUTPUT_DIR)

# ============================================================
# RUN PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("RUNNING PREDICTIONS")
print("=" * 60)

results = model.predict(
    source=str(TEST_IMAGES),
    conf=CONFIDENCE,
    imgsz=640,

    # Save annotated images
    save=True,

    # Save prediction labels as well
    save_txt=True,

    # Save confidence scores
    save_conf=True,

    project=str(OUTPUT_DIR.parent),
    name=OUTPUT_DIR.name,
    exist_ok=True,

    verbose=True
)

# ============================================================
# COUNT CELLS
# ============================================================

total_wbc = 0
total_rbc = 0
total_platelets = 0

print()
print("=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

for i, result in enumerate(results):

    wbc = 0
    rbc = 0
    platelets = 0

    if result.boxes is not None:

        for cls in result.boxes.cls:

            class_id = int(cls)

            # CORRECT MAPPING
            if class_id == 0:
                platelets += 1

            elif class_id == 1:
                rbc += 1

            elif class_id == 2:
                wbc += 1

    total_wbc += wbc
    total_rbc += rbc
    total_platelets += platelets

    print(
        f"{i + 1:3}/{len(results)}  "
        f"WBC={wbc:2}  "
        f"RBC={rbc:3}  "
        f"Platelets={platelets:2}"
    )

# ============================================================
# FINAL TOTALS
# ============================================================

print()
print("=" * 60)
print("TOTAL PREDICTIONS")
print("=" * 60)

print(f"WBC        : {total_wbc}")
print(f"RBC        : {total_rbc}")
print(f"Platelets  : {total_platelets}")

print()
print("=" * 60)
print("OUTPUT")
print("=" * 60)

print()
print("Annotated images:")
print(OUTPUT_DIR)

print()
print("Prediction labels:")
print(OUTPUT_DIR / "labels")

print()
print("Confidence threshold:")
print(CONFIDENCE)

print()
print("Done!")