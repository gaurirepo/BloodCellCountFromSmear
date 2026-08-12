from ultralytics import YOLO
import cv2
import os


# ============================================================
# SETTINGS
# ============================================================

MODEL = "model.pt"

IMAGE_FOLDER = "DataSet/images/test"
LABEL_FOLDER = "DataSet/labels/test"

OUTPUT_FOLDER = "comparison_images"

CONFIDENCE = 0.50


# ============================================================
# THE 10 WORST IMAGES FROM YOUR TEST
# ============================================================

worst_images = [
    "1299047edeb15e2581d9680aaa667d5d.png",
    "69382cf7216652f69811f8f0078ef688.png",
    "13cd097a957e573b84aee503a173dd94.png",
    "cba1d43b52265420a9ee236096572e1f.png",
    "a2a35a2cd6c857bbb910390bcf92c78a.png",
    "0c4de1877622533984edd9e5e3108a78.png",
    "2eaeaa882c9555ec971c1c85979dd9c0.png",
    "579f9dd49c365f668b1de414493488a2.png",
    "63fc7f6fa96e5f509f2a5849121b9110.png",
    "7f96e833a77559fb8fe76152b2b76660.png"
]


# ============================================================
# CLASS NAMES
# ============================================================

class_names = {
    0: "WBC",
    1: "RBC",
    2: "Platelets"
}


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

print("\nLoading YOLO model...")

model = YOLO(MODEL)

print("Model loaded.")


# ============================================================
# FUNCTION TO READ GROUND TRUTH
# ============================================================

def read_ground_truth(
        image_file,
        image_width,
        image_height
):

    base_name = os.path.splitext(
        image_file
    )[0]

    label_file = os.path.join(
        LABEL_FOLDER,
        base_name + ".txt"
    )

    boxes = []


    if not os.path.exists(label_file):

        return boxes


    with open(
            label_file,
            "r"
    ) as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) < 5:

                continue


            class_id = int(
                parts[0]
            )

            # YOLO format:
            # class x_center y_center width height
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])


            # Convert normalized coordinates
            # to pixel coordinates

            x1 = int(
                (x_center - width / 2)
                * image_width
            )

            y1 = int(
                (y_center - height / 2)
                * image_height
            )

            x2 = int(
                (x_center + width / 2)
                * image_width
            )

            y2 = int(
                (y_center + height / 2)
                * image_height
            )


            boxes.append(
                (
                    class_id,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


    return boxes


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_number, image_file in enumerate(
        worst_images,
        start=1
):

    print(
        f"\nProcessing "
        f"{image_number}/10: "
        f"{image_file}"
    )


    image_path = os.path.join(
        IMAGE_FOLDER,
        image_file
    )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            "Could not open image."
        )

        continue


    height, width = image.shape[:2]


    # ========================================================
    # DRAW GROUND TRUTH
    # GREEN = ACTUAL CELLS
    # ========================================================

    ground_truth_boxes = read_ground_truth(
        image_file,
        width,
        height
    )


    for (
            class_id,
            x1,
            y1,
            x2,
            y2
    ) in ground_truth_boxes:


        # GREEN
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            image,
            "GT " + class_names[class_id],
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )


    # ========================================================
    # RUN YOLO
    # ========================================================

    result = model.predict(
        image_path,
        conf=CONFIDENCE,
        verbose=False
    )[0]


    # ========================================================
    # DRAW YOLO PREDICTIONS
    # RED = AI PREDICTION
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            x1, y1, x2, y2 = coordinates


            # RED
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )


            cv2.putText(
                image,
                "AI "
                + class_names[class_id]
                + f" {confidence:.2f}",
                (x1, min(y2 + 15, height - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )


    # ========================================================
    # ADD LEGEND
    # ========================================================

    cv2.rectangle(
        image,
        (5, 5),
        (330, 55),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        image,
        "GREEN = Ground Truth",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


    cv2.putText(
        image,
        "RED = AI Prediction",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = os.path.join(
        OUTPUT_FOLDER,
        image_file
    )


    cv2.imwrite(
        output_path,
        image
    )


print("\n========================================")
print("DONE!")
print("========================================")

print(
    f"\nComparison images are in:"
)

print(
    f"{OUTPUT_FOLDER}/"
)