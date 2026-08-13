from ultralytics import YOLO

# Load the best model from the original full-dataset training
model = YOLO("runs/yolo26/full_training/weights/best.pt")

# Fine-tune using the manually corrected dataset
model.train(
    data="CorrectedDataSet_20/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,

    project="runs/yolo26",
    name="corrected_20_retrain"
)

print("\nRetraining complete.")
print("New model:")
print("runs/yolo26/corrected_20_retrain/weights/best.pt")