from ultralytics import YOLO
import os
import cv2


# ============================================================
# SETTINGS
# ============================================================

PROJECT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_FOLDER,
    "model.pt"
)

DATASET = os.path.join(
    PROJECT_FOLDER,
    "DataSet_TXL_PBC"
)

TEST_IMAGES = os.path.join(
    DATASET,
    "images",
    "test"
)

TEST_LABELS = os.path.join(
    DATASET,
    "labels",
    "test"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "rbc_visual_audit"
)

CONFIDENCE = 0.50


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==============================================")
print("RBC VISUAL AUDIT")
print("==============================================")

print("\nLoading model...")

model = YOLO(MODEL_PATH)

print("Model loaded!")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "WBC",
    1: "RBC",
    2: "Platelets"
}


# ============================================================
# GET TEST IMAGES
# ============================================================

image_files = [
    f
    for f in os.listdir(TEST_IMAGES)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

image_files.sort()

print(
    f"\nTesting {len(image_files)} images..."
)


# ============================================================
# STORE IMAGE ERRORS
# ============================================================

image_results = []


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, image_file in enumerate(
        image_files,
        start=1
):

    image_path = os.path.join(
        TEST_IMAGES,
        image_file
    )

    label_file = (
            os.path.splitext(image_file)[0]
            + ".txt"
    )

    label_path = os.path.join(
        TEST_LABELS,
        label_file
    )


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:
        continue


    height, width = image.shape[:2]


    # --------------------------------------------------------
    # READ GROUND TRUTH
    # --------------------------------------------------------

    ground_truth_rbc = []


    if os.path.exists(label_path):

        with open(
                label_path,
                "r"
        ) as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue


                class_id = int(parts[0])


                if class_id != 1:
                    continue


                # YOLO format:
                # class x_center y_center width height

                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height


                x1 = int(
                    x_center - box_width / 2
                )

                y1 = int(
                    y_center - box_height / 2
                )

                x2 = int(
                    x_center + box_width / 2
                )

                y2 = int(
                    y_center + box_height / 2
                )


                ground_truth_rbc.append(
                    (x1, y1, x2, y2)
                )


    # --------------------------------------------------------
    # RUN YOLO
    # --------------------------------------------------------

    results = model.predict(
        source=image_path,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]


    predicted_rbc = []


    if result.boxes is not None:

        for i in range(
                len(result.boxes)
        ):

            class_id = int(
                result.boxes.cls[i].item()
            )


            if class_id != 1:
                continue


            confidence = float(
                result.boxes.conf[i].item()
            )


            box = result.boxes.xyxy[i].cpu().numpy()

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])


            predicted_rbc.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence
                )
            )


    # --------------------------------------------------------
    # CALCULATE ERROR
    # --------------------------------------------------------

    actual_count = len(
        ground_truth_rbc
    )

    ai_count = len(
        predicted_rbc
    )

    difference = (
            ai_count - actual_count
    )

    absolute_error = abs(
        difference
    )


    image_results.append(
        (
            absolute_error,
            image_file,
            actual_count,
            ai_count,
            difference
        )
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 10 == 0:

        print(
            f"Processed "
            f"{index}/{len(image_files)}..."
        )


# ============================================================
# SORT BY WORST RBC ERROR
# ============================================================

image_results.sort(
    reverse=True
)


# ============================================================
# SELECT TOP 10
# ============================================================

worst_images = image_results[:10]


print("\n==============================================")
print("TOP 10 RBC COUNTING ERRORS")
print("==============================================")

print(
    f"\n{'Image':<45}"
    f"{'Actual':>8}"
    f"{'AI':>8}"
    f"{'Error':>8}"
)


for (
        error,
        image_file,
        actual_count,
        ai_count,
        difference
) in worst_images:

    print(
        f"{image_file:<45}"
        f"{actual_count:>8}"
        f"{ai_count:>8}"
        f"{difference:>+8}"
    )


# ============================================================
# CREATE VISUALIZATIONS
# ============================================================

print("\nCreating visual audit images...")


for rank, (
        error,
        image_file,
        actual_count,
        ai_count,
        difference
) in enumerate(
    worst_images,
    start=1
):

    image_path = os.path.join(
        TEST_IMAGES,
        image_file
    )

    label_file = (
            os.path.splitext(image_file)[0]
            + ".txt"
    )

    label_path = os.path.join(
        TEST_LABELS,
        label_file
    )


    image = cv2.imread(
        image_path
    )


    if image is None:
        continue


    height, width = image.shape[:2]


    # --------------------------------------------------------
    # DRAW GROUND TRUTH
    # GREEN = TRUE LABEL
    # --------------------------------------------------------

    if os.path.exists(label_path):

        with open(
                label_path,
                "r"
        ) as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue


                class_id = int(parts[0])


                if class_id != 1:
                    continue


                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height


                x1 = int(
                    x_center - box_width / 2
                )

                y1 = int(
                    y_center - box_height / 2
                )

                x2 = int(
                    x_center + box_width / 2
                )

                y2 = int(
                    y_center + box_height / 2
                )


                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    image,
                    "GT RBC",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1
                )


    # --------------------------------------------------------
    # RUN MODEL AGAIN
    # --------------------------------------------------------

    results = model.predict(
        source=image_path,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # DRAW AI PREDICTIONS
    # RED = AI
    # --------------------------------------------------------

    if result.boxes is not None:

        for i in range(
                len(result.boxes)
        ):

            class_id = int(
                result.boxes.cls[i].item()
            )


            if class_id != 1:
                continue


            confidence = float(
                result.boxes.conf[i].item()
            )


            box = result.boxes.xyxy[i].cpu().numpy()


            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])


            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )


            cv2.putText(
                image,
                f"AI RBC {confidence:.2f}",
                (x1, min(height - 5, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 255),
                1
            )


    # --------------------------------------------------------
    # ADD HEADER
    # --------------------------------------------------------

    header_height = 70

    canvas = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )


    cv2.putText(
        canvas,
        f"RBC AUDIT #{rank}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 0, 0),
        2
    )


    cv2.putText(
        canvas,
        f"Ground Truth: {actual_count}    "
        f"AI: {ai_count}    "
        f"Difference: {difference:+d}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 0),
        2
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_name = (
        f"{rank:02d}_"
        f"{os.path.splitext(image_file)[0]}.png"
    )


    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )


    cv2.imwrite(
        output_path,
        canvas
    )


# ============================================================
# FINISH
# ============================================================

print("\n==============================================")
print("VISUAL AUDIT COMPLETE")
print("==============================================")

print("\nResults saved to:")

print(
    OUTPUT_FOLDER
)

print(
    "\nGREEN boxes = Ground Truth"
)

print(
    "RED boxes = AI predictions"
)

print(
    "\nOpen the images in:"
)

print(
    "rbc_visual_audit"
)

print(
    "\n=============================================="
)