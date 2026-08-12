from ultralytics import YOLO
from collections import Counter
import os


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO("model.pt")


# ============================================================
# DATASET
# ============================================================

image_folder = "DataSet/images/test"
label_folder = "DataSet/labels/test"


class_names = {
    0: "WBC",
    1: "RBC",
    2: "Platelets"
}


# ============================================================
# TEST DIFFERENT CONFIDENCE LEVELS
# ============================================================

confidence_levels = [
    0.50,
    0.40,
    0.30,
    0.20
]


# ============================================================
# GET IMAGES
# ============================================================

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]


print("\n========================================")
print("CONFIDENCE THRESHOLD EXPERIMENT")
print("========================================")

print(
    f"\nTesting {len(image_files)} images."
)


# ============================================================
# TEST EACH CONFIDENCE LEVEL
# ============================================================

for confidence in confidence_levels:

    total_actual = Counter()
    total_predicted = Counter()
    total_error = Counter()

    exact_matches = Counter()


    print(
        f"\n\n----------------------------------------"
    )

    print(
        f"CONFIDENCE = {confidence}"
    )

    print(
        "----------------------------------------"
    )


    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image_file in image_files:

        image_path = os.path.join(
            image_folder,
            image_file
        )


        # ====================================================
        # GROUND TRUTH
        # ====================================================

        base_name = os.path.splitext(
            image_file
        )[0]

        label_file = os.path.join(
            label_folder,
            base_name + ".txt"
        )


        actual = Counter()


        if os.path.exists(label_file):

            with open(
                    label_file,
                    "r"
            ) as file:

                for line in file:

                    parts = line.strip().split()

                    if len(parts) >= 5:

                        class_id = int(parts[0])

                        actual[
                            class_names[class_id]
                        ] += 1


        # ====================================================
        # AI PREDICTION
        # ====================================================

        result = model.predict(
            image_path,
            conf=confidence,
            verbose=False
        )[0]


        predicted = Counter()


        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                predicted[
                    class_names[class_id]
                ] += 1


        # ====================================================
        # SAVE TOTALS
        # ====================================================

        for cell_type in [
            "WBC",
            "RBC",
            "Platelets"
        ]:

            actual_count = actual[
                cell_type
            ]

            predicted_count = predicted[
                cell_type
            ]

            total_actual[
                cell_type
            ] += actual_count

            total_predicted[
                cell_type
            ] += predicted_count

            total_error[
                cell_type
            ] += abs(
                actual_count -
                predicted_count
            )

            if actual_count == predicted_count:

                exact_matches[
                    cell_type
                ] += 1


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        f"\n{'Cell':<15}"
        f"{'Actual':>10}"
        f"{'AI':>10}"
        f"{'Difference':>12}"
        f"{'MAE':>10}"
        f"{'Exact %':>10}"
    )

    print("-" * 67)


    for cell_type in [
        "WBC",
        "RBC",
        "Platelets"
    ]:

        actual = total_actual[
            cell_type
        ]

        predicted = total_predicted[
            cell_type
        ]

        difference = (
                predicted - actual
        )

        mae = (
                total_error[cell_type]
                / len(image_files)
        )

        exact = (
                exact_matches[cell_type]
                / len(image_files)
                * 100
        )


        print(
            f"{cell_type:<15}"
            f"{actual:>10}"
            f"{predicted:>10}"
            f"{difference:>+12}"
            f"{mae:>10.2f}"
            f"{exact:>9.2f}%"
        )


print("\n========================================")
print("EXPERIMENT COMPLETE")
print("========================================")