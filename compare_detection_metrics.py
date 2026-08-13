from ultralytics import YOLO
from pathlib import Path
import yaml

OLD_MODEL = "runs/yolo26/full_training/weights/best.pt"

NEW_MODEL = (
    "runs/detect/runs/yolo26/"
    "corrected_20_retrain-3/weights/best.pt"
)

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_YAML = PROJECT_ROOT / "test_only.yaml"


# ------------------------------------------------------------
# CREATE TEST-ONLY DATASET YAML
# ------------------------------------------------------------

test_config = {
    "path": str(PROJECT_ROOT / "Dataset"),
    "train": "test/images",
    "val": "test/images",
    "test": "test/images",
    "names": {
        0: "Platelets",
        1: "RBC",
        2: "WBC"
    }
}

with open(TEST_YAML, "w") as f:
    yaml.safe_dump(test_config, f, sort_keys=False)


def evaluate_model(model_path, name):

    print("\n")
    print("=" * 80)
    print(name)
    print("=" * 80)

    model = YOLO(model_path)

    metrics = model.val(
        data=str(TEST_YAML),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        plots=True,
        verbose=True
    )

    print("\nSUMMARY")
    print("-" * 60)

    print(f"mAP50       : {metrics.box.map50:.4f}")
    print(f"mAP50-95    : {metrics.box.map:.4f}")
    print(f"Mean Precision: {metrics.box.mp:.4f}")
    print(f"Mean Recall   : {metrics.box.mr:.4f}")

    return metrics


old_metrics = evaluate_model(
    OLD_MODEL,
    "ORIGINAL MODEL"
)

new_metrics = evaluate_model(
    NEW_MODEL,
    "RETRAINED MODEL"
)


print("\n\n")
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(
    f"{'Metric':<20}"
    f"{'Original':>15}"
    f"{'Retrained':>15}"
    f"{'Difference':>15}"
)

print("-" * 80)

metrics_to_compare = [
    (
        "Precision",
        old_metrics.box.mp,
        new_metrics.box.mp
    ),
    (
        "Recall",
        old_metrics.box.mr,
        new_metrics.box.mr
    ),
    (
        "mAP50",
        old_metrics.box.map50,
        new_metrics.box.map50
    ),
    (
        "mAP50-95",
        old_metrics.box.map,
        new_metrics.box.map
    )
]

for label, old, new in metrics_to_compare:

    diff = new - old

    print(
        f"{label:<20}"
        f"{old:>15.4f}"
        f"{new:>15.4f}"
        f"{diff:>+15.4f}"
    )

print("=" * 80)