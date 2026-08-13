from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# START FROM OUR CORRECTED / FINE-TUNED MODEL
# ------------------------------------------------------------

MODEL_PATH = (
        PROJECT_DIR
        / "runs"
        / "detect"
        / "runs"
        / "yolo26"
        / "corrected_20_retrain-3"
        / "weights"
        / "best.pt"
)

# ------------------------------------------------------------
# FULL DATASET
# ------------------------------------------------------------

DATA_YAML = PROJECT_DIR / "data.yaml"

print("=" * 70)
print("FULL DATASET RETRAINING")
print("=" * 70)

print(f"\nStarting model : {MODEL_PATH}")
print(f"Dataset        : {DATA_YAML}")

model = YOLO(str(MODEL_PATH))

model.train(
    data=str(DATA_YAML),

    # Training
    epochs=30,
    imgsz=640,
    batch=8,

    # Fine-tuning rather than aggressive re-learning
    lr0=0.001,

    # Keep output separate
    project="runs/yolo26",
    name="full_training_from_corrected",

    patience=10,
    plots=True
)

print("\n" + "=" * 70)
print("FULL RETRAINING COMPLETE")
print("=" * 70)