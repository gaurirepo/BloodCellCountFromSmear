from ultralytics import YOLO
from pathlib import Path

print("=" * 60)
print("YOLO26 - QUICK TEST TRAINING")
print("=" * 60)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_YAML = "data.yaml"

# YOLO26 nano model
MODEL_NAME = "yolo26n.pt"

# Small quick experiment
EPOCHS = 3
IMAGE_SIZE = 640

# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

print("\nChecking dataset...")

for folder in [
    "Dataset/train/images",
    "Dataset/train/labels",
    "Dataset/valid/images",
    "Dataset/valid/labels",
    "Dataset/test/images",
    "Dataset/test/labels"
]:
    path = Path(folder)

    if not path.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    print(f"OK: {folder}")

# --------------------------------------------------
# COUNT IMAGES
# --------------------------------------------------

train_images = len(list(Path("Dataset/train/images").glob("*")))
val_images = len(list(Path("Dataset/valid/images").glob("*")))
test_images = len(list(Path("Dataset/test/images").glob("*")))

print("\nDataset:")
print(f"Train images : {train_images}")
print(f"Val images   : {val_images}")
print(f"Test images  : {test_images}")

# --------------------------------------------------
# LOAD YOLO26
# --------------------------------------------------

print("\nLoading YOLO26n...")

model = YOLO(MODEL_NAME)

print("Model loaded successfully!")

print("\nModel classes:")
print(model.names)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nStarting QUICK training...")
print(f"Epochs: {EPOCHS}")
print(f"Image size: {IMAGE_SIZE}")

results = model.train(

    data=DATA_YAML,

    epochs=EPOCHS,

    imgsz=IMAGE_SIZE,

    # Use CPU for IntelliJ/Mac
    # Change to 0 if using NVIDIA GPU
    device="cpu",

    project="runs/yolo26",

    name="quick_test",

    exist_ok=True,

    patience=3,

    workers=2,

    seed=42,

    verbose=True
)

print("\n" + "=" * 60)
print("QUICK TRAINING COMPLETE")
print("=" * 60)

print("\nModel should be here:")

print(
    "runs/yolo26/quick_test/weights/model.pt"
)