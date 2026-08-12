import os
import shutil


# ============================================================
# ORIGINAL AND CLEAN DATASET
# ============================================================

SOURCE = "TXL-PBC"
DESTINATION = "TXL-PBC-CLEAN"


SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# HEADER
# ============================================================

print("\n==============================================")
print("CREATING CLEAN DATASET")
print("==============================================")

print("\nOriginal dataset:")
print(SOURCE)

print("\nClean dataset:")
print(DESTINATION)


# ============================================================
# CREATE DATASET
# ============================================================

if os.path.exists(DESTINATION):

    print(
        "\nWARNING: Clean dataset already exists."
    )

    print(
        "Delete it manually if you want to recreate it."
    )

    exit()


os.makedirs(
    DESTINATION
)


# ============================================================
# COPY data.yaml
# ============================================================

source_yaml = os.path.join(
    SOURCE,
    "data.yaml"
)

destination_yaml = os.path.join(
    DESTINATION,
    "data.yaml"
)


if os.path.exists(source_yaml):

    shutil.copy2(
        source_yaml,
        destination_yaml
    )


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

total_removed = 0


for split in SPLITS:

    print(
        f"\nProcessing {split}..."
    )


    source_images = os.path.join(
        SOURCE,
        "images",
        split
    )

    source_labels = os.path.join(
        SOURCE,
        "labels",
        split
    )


    destination_images = os.path.join(
        DESTINATION,
        "images",
        split
    )

    destination_labels = os.path.join(
        DESTINATION,
        "labels",
        split
    )


    os.makedirs(
        destination_images,
        exist_ok=True
    )

    os.makedirs(
        destination_labels,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    image_files = [
        f for f in os.listdir(source_images)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]


    for image_file in image_files:

        shutil.copy2(
            os.path.join(
                source_images,
                image_file
            ),
            os.path.join(
                destination_images,
                image_file
            )
        )


    # --------------------------------------------------------
    # Clean labels
    # --------------------------------------------------------

    label_files = [
        f for f in os.listdir(source_labels)
        if f.endswith(".txt")
    ]


    split_removed = 0


    for label_file in label_files:

        source_label = os.path.join(
            source_labels,
            label_file
        )

        destination_label = os.path.join(
            destination_labels,
            label_file
        )


        with open(
                source_label,
                "r"
        ) as file:

            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]


        # ----------------------------------------------------
        # Remove EXACT duplicate lines
        # ----------------------------------------------------

        unique_lines = []

        seen = set()


        for line in lines:

            if line not in seen:

                unique_lines.append(
                    line
                )

                seen.add(line)

            else:

                split_removed += 1
                total_removed += 1


        # ----------------------------------------------------
        # Write cleaned label
        # ----------------------------------------------------

        with open(
                destination_label,
                "w"
        ) as file:

            for line in unique_lines:

                file.write(
                    line + "\n"
                )


    print(
        f"  Images copied : "
        f"{len(image_files)}"
    )

    print(
        f"  Labels copied : "
        f"{len(label_files)}"
    )

    print(
        f"  Duplicates removed : "
        f"{split_removed}"
    )


# ============================================================
# FINISH
# ============================================================

print("\n==============================================")
print("CLEAN DATASET CREATED")
print("==============================================")

print(
    f"\nTotal duplicate annotations removed: "
    f"{total_removed}"
)

print(
    f"\nClean dataset location:"
)

print(
    DESTINATION
)

print(
    "\nOriginal TXL-PBC dataset was NOT modified."
)

print(
    "=============================================="
)