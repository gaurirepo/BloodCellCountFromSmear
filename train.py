from ultralytics import YOLO
from pathlib import Path

print("=" * 60)
print("YOLO26 - FULL BLOOD CELL TRAINING")
print("=" * 60)

# --------------------------------------------------
# PROJECT LOCATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_YAML = PROJECT_DIR / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"\ndata.yaml not found:\n{DATA_YAML}"
    )

print("\nProject:")
print(PROJECT_DIR)

print("\nDataset:")
print(DATA_YAML)

# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

folders = [
    PROJECT_DIR / "Dataset/train/images",
    PROJECT_DIR / "Dataset/train/labels",
    PROJECT_DIR / "Dataset/valid/images",
    PROJECT_DIR / "Dataset/valid/labels",
    PROJECT_DIR / "Dataset/test/images",
    PROJECT_DIR / "Dataset/test/labels"
]

print("\nChecking dataset...")

for folder in folders:

    if not folder.exists():
        raise FileNotFoundError(
            f"\nMissing:\n{folder}"
        )

    print(f"OK: {folder}")

# --------------------------------------------------
# LOAD YOLO26
# --------------------------------------------------

print("\nLoading YOLO26n...")

model = YOLO("yolo26n.pt")

print("YOLO26n loaded successfully!")

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nStarting FULL TRAINING")

print("Epochs : 50")
print("Image size : 640")
print("Device : CPU")

results = model.train(

    data=str(DATA_YAML),

    epochs=50,

    imgsz=640,

    device="cpu",

    project=str(PROJECT_DIR / "runs/yolo26"),

    name="full_training",

    exist_ok=True,

    patience=10,

    workers=2,

    seed=42,

    verbose=True
)

# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n" + "=" * 60)
print("FULL YOLO26 TRAINING COMPLETE")
print("=" * 60)

print("\nBest Trained model:")

print(
    PROJECT_DIR /
    "runs/detect/runs/yolo26/full_training/weights/trainedmodel.pt"
)

print("\nLast model:")

print(
    PROJECT_DIR /
    "runs/detect/runs/yolo26/full_training/weights/last.pt"
)