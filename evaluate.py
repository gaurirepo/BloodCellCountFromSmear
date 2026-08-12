from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = (
    "runs/detect/blood_cell_detector/weights/model.pt"
)

DATA_YAML = "TXL-PBC/data.yaml"


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading trained model..."
)

model = YOLO(
    MODEL_PATH
)


# ============================================================
# VALIDATE
# ============================================================

print(
    "\nEvaluating on TEST dataset..."
)


metrics = model.val(

    data=DATA_YAML,

    split="test",

    imgsz=640,

    batch=8,

    plots=True,

    verbose=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================\n")


print(
    "mAP@50:",
    metrics.box.map50
)


print(
    "mAP@50-95:",
    metrics.box.map
)


print(
    "Precision:",
    metrics.box.mp
)


print(
    "Recall:",
    metrics.box.mr
)


print("\nEvaluation complete!")