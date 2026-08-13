from ultralytics import YOLO
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
        PROJECT_DIR
        / "runs/yolo26/full_training/weights/best.pt"
)

# CHANGE THIS to the image you want to test
IMAGE_PATH = (
        PROJECT_DIR
        / "Dataset/test/images/BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg"
)

OUTPUT_DIR = PROJECT_DIR / "runs/yolo26_evaluation/single_image_test"

IMAGE_SIZE = 640

# Test several confidence thresholds
CONFIDENCE_LEVELS = [0.25, 0.35, 0.50, 0.65]

# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("BLOOD CELL SINGLE IMAGE MODEL TEST")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nImage:")
print(IMAGE_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fully trained model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")

print("\nModel classes:")
print(model.names)

# ============================================================
# TEST EACH CONFIDENCE THRESHOLD
# ============================================================

for conf in CONFIDENCE_LEVELS:

    print("\n")
    print("=" * 70)
    print(f"TESTING CONFIDENCE = {conf}")
    print("=" * 70)

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=conf,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False,
        save=True,
        project=str(OUTPUT_DIR),
        name=f"conf_{str(conf).replace('.', '_')}",
        exist_ok=True
    )

    result = results[0]

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    counts = {
        "Platelets": 0,
        "RBC": 0,
        "WBC": 0
    }

    print("\nDETECTIONS")
    print("-" * 70)

    if result.boxes is None or len(result.boxes) == 0:

        print("NO DETECTIONS")

    else:

        for i, box in enumerate(result.boxes):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            counts[class_name] += 1

            print(
                f"{i + 1:02d}. "
                f"{class_name:10s} "
                f"confidence={confidence:.3f} "
                f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nCOUNTS")
    print("-" * 70)

    print(f"Platelets : {counts['Platelets']}")
    print(f"RBC       : {counts['RBC']}")
    print(f"WBC       : {counts['WBC']}")
    print(
        f"Total     : "
        f"{counts['Platelets'] + counts['RBC'] + counts['WBC']}"
    )

    print("\nSaved annotated image to:")
    print(
        OUTPUT_DIR
        / f"conf_{str(conf).replace('.', '_')}"
    )

print("\n")
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)