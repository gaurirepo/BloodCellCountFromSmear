from ultralytics import YOLO
from collections import Counter
import os


# ============================================================
# MODEL
# ============================================================

model = YOLO("model.pt")


# ============================================================
# IMAGE WE ARE INVESTIGATING
# ============================================================

image = "TXL-PBC/images/test/1299047edeb15e2581d9680aaa667d5d.png"


# ============================================================
# CONFIDENCE LEVELS
# ============================================================

confidence_levels = [
    0.50,
    0.40,
    0.30,
    0.20,
    0.10
]


print("\n========================================")
print("SINGLE IMAGE RBC INVESTIGATION")
print("========================================")

print("\nGround truth RBC count = 22")


for confidence in confidence_levels:

    result = model.predict(
        image,
        conf=confidence,
        verbose=False
    )[0]


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


    print("\n----------------------------------------")

    print(
        f"Confidence threshold: {confidence}"
    )

    print("----------------------------------------")

    print(
        "WBC       :",
        counts["WBC"]
    )

    print(
        "RBC       :",
        counts["RBC"]
    )

    print(
        "Platelets :",
        counts["Platelets"]
    )

    print(
        "TOTAL     :",
        sum(counts.values())
    )


print("\n========================================")
print("INVESTIGATION COMPLETE")
print("========================================")