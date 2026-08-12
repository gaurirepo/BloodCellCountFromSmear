import os
import yaml
from collections import Counter


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "DataSet_TXL_PBC"


# ============================================================
# COUNT LABELS
# ============================================================

def count_labels(label_directory):

    counts = Counter()

    if not os.path.exists(label_directory):

        print(
            f"Directory not found: {label_directory}"
        )

        return counts

    for filename in os.listdir(label_directory):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(
            label_directory,
            filename
        )

        with open(filepath, "r") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                class_id = int(parts[0])

                counts[class_id] += 1

    return counts


# ============================================================
# START
# ============================================================

print("\n==============================")
print("BLOOD CELL DATASET CHECK")
print("==============================\n")


# ============================================================
# FIND DATA.YAML
# ============================================================

yaml_path = os.path.join(
    DATASET_DIR,
    "data.yaml"
)


if not os.path.exists(yaml_path):

    print("ERROR:")
    print(
        f"Could not find: {yaml_path}"
    )

    print("\nMake sure your project looks like:")

    print("""
CountBloodCells/
│
├── check_dataset.py
├── train.py
├── predict.py
├── evaluate.py
├── app.py
│
└── TXL-PBC/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    └── labels/
        ├── train/
        ├── val/
        └── test/
""")

    exit()


# ============================================================
# READ DATA.YAML
# ============================================================

with open(
        yaml_path,
        "r"
) as file:

    data = yaml.safe_load(file)


print("Dataset configuration:")

print(data)


# ============================================================
# READ CLASS NAMES
# ============================================================

names = data.get(
    "names",
    []
)


print("\nClasses:")


# Handle list format:
#
# names:
#   - WBC
#   - RBC
#   - Platelets

if isinstance(names, list):

    class_names = {
        index: name
        for index, name
        in enumerate(names)
    }


# Handle dictionary format:
#
# names:
#   0: WBC
#   1: RBC
#   2: Platelets

elif isinstance(names, dict):

    class_names = {
        int(class_id): name
        for class_id, name
        in names.items()
    }


else:

    print(
        "ERROR: Could not understand "
        "the class names in data.yaml."
    )

    exit()


for class_id, name in class_names.items():

    print(
        f"{class_id}: {name}"
    )


# ============================================================
# CHECK NUMBER OF CLASSES
# ============================================================

expected_classes = 3

if len(class_names) != expected_classes:

    print(
        f"\nWARNING: Expected {expected_classes} "
        f"classes but found {len(class_names)}."
    )


# ============================================================
# COUNT TRAIN / VAL / TEST
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    label_directory = os.path.join(
        DATASET_DIR,
        "labels",
        split
    )

    image_directory = os.path.join(
        DATASET_DIR,
        "images",
        split
    )


    # --------------------------------------------------------
    # Count labels
    # --------------------------------------------------------

    label_counts = count_labels(
        label_directory
    )


    # --------------------------------------------------------
    # Count images
    # --------------------------------------------------------

    image_count = 0


    if os.path.exists(
            image_directory
    ):

        image_count = len([

            f

            for f in os.listdir(
                image_directory
            )

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                )
            )

        ])


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n------------------------------")

    print(
        split.upper()
    )

    print("------------------------------")


    print(
        "Images:",
        image_count
    )


    total_objects = sum(
        label_counts.values()
    )


    print(
        "Objects:",
        total_objects
    )


    for class_id in sorted(
            class_names.keys()
    ):

        class_name = class_names[
            class_id
        ]

        count = label_counts.get(
            class_id,
            0
        )


        print(
            f"{class_name:12s}: {count}"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n==============================")

print(
    "DATASET CHECK COMPLETE!"
)

print("==============================\n")