from ultralytics import YOLO
from collections import Counter
import os


# Load our trained model
model = YOLO("model.pt")


# Test images are here
test_folder = "DataSet/images/test"


# Find images
images = [
    f for f in os.listdir(test_folder)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


print("\n==============================")
print("BLOOD CELL DETECTOR")
print("==============================")

print(
    f"\nFound {len(images)} test images."
)


# Use the first test image
image = os.path.join(
    test_folder,
    images[0]
)


print(
    f"\nTesting image: {images[0]}"
)


# Run model
results = model.predict(
    image,
    conf=0.5,
    save=True
)


# Get result
result = results[0]


# Count cells
counts = Counter()


if result.boxes is not None:

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        class_name = model.names[
            class_id
        ]

        counts[class_name] += 1


# Display results
print("\n==============================")
print("RESULT")
print("==============================")

print(
    "WBC       :",
    counts.get("WBC", 0)
)

print(
    "RBC       :",
    counts.get("RBC", 0)
)

print(
    "Platelets :",
    counts.get("Platelets", 0)
)

print(
    "------------------------------"
)

print(
    "TOTAL     :",
    sum(counts.values())
)

print("\nDone!")

print(
    "\nThe annotated image has been "
    "saved by YOLO in the runs folder."
)