from ultralytics import YOLO
from pathlib import Path
import os

# ============================================================
# YOLO26 QUICK TEST MODEL - ACCURACY + PREDICTION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_PATH = (
        PROJECT_DIR
        / "runs"
        / "detect"
        / "runs"
        / "yolo26"
        / "quick_test"
        / "weights"
        / "best.pt"
)

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATA_YAML = PROJECT_DIR / "data.yaml"

TEST_IMAGES = (
        PROJECT_DIR
        / "Dataset"
        / "test"
        / "images"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = (
        PROJECT_DIR
        / "runs"
        / "yolo26_evaluation"
        / "quick_test"
)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = 640

# For standard YOLO validation, keep this very low.
# This allows YOLO to calculate its full precision/recall curve.
VALIDATION_CONF = 0.001

# For actual displayed predictions/counting.
PREDICTION_CONF = 0.50

# ============================================================
# CORRECT DATASET CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC"
}

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("YOLO26 QUICK TEST MODEL - ACCURACY + PREDICTION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nDataset:")
print(DATA_YAML)

print("\nTest images:")
print(TEST_IMAGES)

print("\nExpected class mapping:")
for class_id, class_name in CLASS_NAMES.items():
    print(f"  {class_id} -> {class_name}")

# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "-" * 70)
print("CHECKING FILES")
print("-" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n\n"
        "Check that quick training created:\n"
        "runs/yolo26/quick_test/weights/best.pt"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"\ndata.yaml not found:\n{DATA_YAML}"
    )

if not TEST_IMAGES.exists():
    raise FileNotFoundError(
        f"\nTest image folder not found:\n{TEST_IMAGES}"
    )

print("Model       : OK")
print("data.yaml   : OK")
print("Test images : OK")

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "-" * 70)
print("LOADING MODEL")
print("-" * 70)

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")

print("\nNames stored inside model:")
print(model.names)

# ============================================================
# VERIFY MODEL CLASS NAMES
# ============================================================

print("\nExpected names based on dataset:")
print(CLASS_NAMES)

if model.names != CLASS_NAMES:

    print("\nWARNING!")
    print("The class names stored inside the trained model")
    print("do NOT match the corrected dataset mapping.")
    print()
    print("Model names:")
    print(model.names)
    print()
    print("Expected:")
    print(CLASS_NAMES)
    print()
    print("This is important for interpreting annotated images.")
    print("The model should ideally be retrained after fixing")
    print("data.yaml so the saved model contains the correct names.")

else:

    print("\nClass mapping VERIFIED.")
    print("Model and dataset agree.")

# ============================================================
# COUNT TEST IMAGES
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
    if f.suffix.lower() in image_extensions
]

images.sort()

print("\nNumber of test images:", len(images))

# ============================================================
# YOLO VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TEST SET ACCURACY EVALUATION")
print("=" * 70)

print("\nValidation confidence threshold:", VALIDATION_CONF)
print("Image size:", IMAGE_SIZE)

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=IMAGE_SIZE,
    conf=VALIDATION_CONF,
    plots=True,
    verbose=True,
    project=str(OUTPUT_DIR.parent),
    name="accuracy",
    exist_ok=True
)

# ============================================================
# OVERALL METRICS
# ============================================================

precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)

if precision + recall > 0:

    f1 = (
            2 * precision * recall
            / (precision + recall)
    )

else:

    f1 = 0.0

print("\n" + "=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(f"\nPrecision : {precision:.4f}  ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f}  ({recall * 100:.2f}%)")
print(f"F1        : {f1:.4f}  ({f1 * 100:.2f}%)")
print(f"mAP50     : {map50:.4f}  ({map50 * 100:.2f}%)")
print(f"mAP50-95  : {map5095:.4f}  ({map5095 * 100:.2f}%)")

# ============================================================
# PER CLASS RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS TEST RESULTS")
print("=" * 70)

for class_id in range(len(model.names)):

    class_name = CLASS_NAMES.get(
        class_id,
        str(class_id)
    )

    try:

        p = float(metrics.box.p[class_id])
        r = float(metrics.box.r[class_id])
        ap50 = float(metrics.box.ap50[class_id])
        ap = float(metrics.box.ap[class_id])

        if p + r > 0:

            class_f1 = (
                    2 * p * r
                    / (p + r)
            )

        else:

            class_f1 = 0.0

        print(f"\n{class_name}")

        print(
            f"  Precision : {p:.4f}  "
            f"({p * 100:.2f}%)"
        )

        print(
            f"  Recall    : {r:.4f}  "
            f"({r * 100:.2f}%)"
        )

        print(
            f"  F1        : {class_f1:.4f}  "
            f"({class_f1 * 100:.2f}%)"
        )

        print(
            f"  mAP50     : {ap50:.4f}  "
            f"({ap50 * 100:.2f}%)"
        )

        print(
            f"  mAP50-95  : {ap:.4f}  "
            f"({ap * 100:.2f}%)"
        )

    except Exception as e:

        print(
            f"\nCould not calculate metrics "
            f"for {class_name}"
        )

        print(e)

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 90)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 90)

# ------------------------------------------------------------
# Assessment helper
# ------------------------------------------------------------

def get_assessment(class_name, precision, recall, map50, map5095):

    if class_name == "Platelets":

        if recall < 0.25:
            return "🔴 Very poor"
        elif recall < 0.50:
            return "🟡 Needs improvement"
        elif recall < 0.75:
            return "🟢 Good"
        else:
            return "🟢 Very good"

    elif class_name == "RBC":

        if map50 < 0.50:
            return "🔴 Poor"
        elif map50 < 0.70:
            return "🟡 Reasonable"
        elif map50 < 0.85:
            return "🟢 Good"
        else:
            return "🟢 Very good"

    elif class_name == "WBC":

        if map50 < 0.50:
            return "🔴 Poor"
        elif map50 < 0.75:
            return "🟡 Reasonable"
        elif map50 < 0.90:
            return "🟢 Good"
        else:
            return "🟢 Very good"

    else:

        if map50 < 0.50:
            return "🔴 Poor"
        elif map50 < 0.75:
            return "🟡 Reasonable"
        else:
            return "🟢 Good"


# ------------------------------------------------------------
# Collect results
# ------------------------------------------------------------

summary_rows = []

names = model.names

for i, class_name in names.items():

    precision = float(metrics.box.p[i])
    recall = float(metrics.box.r[i])
    map50 = float(metrics.box.ap50[i])
    map5095 = float(metrics.box.ap[i])

    assessment = get_assessment(
        class_name,
        precision,
        recall,
        map50,
        map5095
    )

    summary_rows.append(
        (
            class_name,
            precision,
            recall,
            map50,
            map5095,
            assessment
        )
    )


# ------------------------------------------------------------
# Overall row
# ------------------------------------------------------------

overall_precision = float(metrics.box.mp)
overall_recall = float(metrics.box.mr)
overall_map50 = float(metrics.box.map50)
overall_map5095 = float(metrics.box.map)

overall_assessment = "🟡 Early-stage"

if overall_map50 >= 0.85:
    overall_assessment = "🟢 Very good"
elif overall_map50 >= 0.75:
    overall_assessment = "🟢 Good"
elif overall_map50 >= 0.60:
    overall_assessment = "🟡 Early-stage"
else:
    overall_assessment = "🔴 Needs improvement"


# ============================================================
# PRINT MARKDOWN TABLE
# ============================================================

print()
print("| Class | Precision | Recall | mAP50 | mAP50-95 | Assessment |")
print("|---|---:|---:|---:|---:|---|")

for (
        class_name,
        precision,
        recall,
        map50,
        map5095,
        assessment
) in summary_rows:

    print(
        f"| **{class_name}** "
        f"| {precision * 100:.2f}% "
        f"| {recall * 100:.2f}% "
        f"| {map50 * 100:.2f}% "
        f"| {map5095 * 100:.2f}% "
        f"| {assessment} |"
    )


print(
    f"| **Overall** "
    f"| {overall_precision * 100:.2f}% "
    f"| {overall_recall * 100:.2f}% "
    f"| {overall_map50 * 100:.2f}% "
    f"| {overall_map5095 * 100:.2f}% "
    f"| {overall_assessment} |"
)

# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TEST IMAGE PREDICTIONS")
print("=" * 70)

print("\nPrediction confidence:", PREDICTION_CONF)

prediction_output = (
        OUTPUT_DIR
        / "predictions"
)

prediction_output.mkdir(
    parents=True,
    exist_ok=True
)

results = model.predict(
    source=str(TEST_IMAGES),
    conf=PREDICTION_CONF,
    imgsz=IMAGE_SIZE,
    save=True,
    project=str(prediction_output.parent),
    name=prediction_output.name,
    exist_ok=True,
    verbose=True
)

# ============================================================
# COUNT PREDICTED CELLS
# ============================================================

total_counts = {
    "Platelets": 0,
    "RBC": 0,
    "WBC": 0
}

print("\n" + "=" * 70)
print("PREDICTED CELL COUNTS")
print("=" * 70)

for i, result in enumerate(results):

    counts = {
        "Platelets": 0,
        "RBC": 0,
        "WBC": 0
    }

    if result.boxes is not None:

        for cls in result.boxes.cls:

            class_id = int(cls)

            class_name = CLASS_NAMES.get(
                class_id,
                f"Unknown_{class_id}"
            )

            if class_name in counts:

                counts[class_name] += 1
                total_counts[class_name] += 1

    print(
        f"{i + 1:3}/{len(results)}  "
        f"WBC={counts['WBC']:2}  "
        f"RBC={counts['RBC']:3}  "
        f"Platelets={counts['Platelets']:2}"
    )

# ============================================================
# TOTAL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("TOTAL PREDICTED CELLS")
print("=" * 70)

print(
    f"WBC        : "
    f"{total_counts['WBC']}"
)

print(
    f"RBC        : "
    f"{total_counts['RBC']}"
)

print(
    f"Platelets  : "
    f"{total_counts['Platelets']}"
)

# ============================================================
# OUTPUT LOCATIONS
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print("\nAccuracy plots:")
print(
    OUTPUT_DIR
    / "accuracy"
)

print("\nAnnotated prediction images:")
print(
    prediction_output
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)