from ultralytics import YOLO
from pathlib import Path

print("=" * 60)
print("YOLO26 BLOOD CELL PREDICTION")
print("=" * 60)

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
        PROJECT_DIR
        / "runs/detect/runs/yolo26/quick_test/weights/best.pt"
)

TEST_FOLDER = PROJECT_DIR / "Dataset/test/images"

OUTPUT_FOLDER = PROJECT_DIR / "runs/yolo26_predictions"

print("\nModel:")
print(MODEL_PATH)

print("\nTest images:")
print(TEST_FOLDER)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_FOLDER.exists():
    raise FileNotFoundError(
        f"Test folder not found:\n{TEST_FOLDER}"
    )

model = YOLO(str(MODEL_PATH))

print("\nModel classes:")
print(model.names)

print("\nRunning predictions...")

results = model.predict(
    source=str(TEST_FOLDER),
    imgsz=640,
    conf=0.5,
    save=True,
    project=str(OUTPUT_FOLDER),
    name="quick_test",
    exist_ok=True
)

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)

print("\nPredicted images saved here:")

print(
    OUTPUT_FOLDER / "quick_test"
)

print("\nOpen that folder and inspect the images.")