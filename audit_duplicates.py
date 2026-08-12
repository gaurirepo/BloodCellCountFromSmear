import os
from collections import Counter


# ============================================================
# DATASET
# ============================================================

DATASET = "TXL-PBC"

SPLITS = [
    "train",
    "val",
    "test"
]


CLASS_NAMES = {
    0: "WBC",
    1: "RBC",
    2: "Platelets"
}


# ============================================================
# HEADER
# ============================================================

print("\n==============================================")
print("DUPLICATE ANNOTATION AUDIT")
print("==============================================")


# IMPORTANT:
# Initialize BOTH counters

total_duplicate_lines = 0
total_duplicate_files = 0


# ============================================================
# CHECK EACH SPLIT
# ============================================================

for split in SPLITS:

    label_folder = os.path.join(
        DATASET,
        "labels",
        split
    )

    print("\n")
    print("==============================================")
    print(f"{split.upper()}")
    print("==============================================")


    if not os.path.exists(label_folder):

        print(
            "Folder not found:",
            label_folder
        )

        continue


    label_files = [
        f for f in os.listdir(label_folder)
        if f.endswith(".txt")
    ]


    split_duplicate_files = 0
    split_duplicate_lines = 0

    split_total_lines = 0
    split_unique_lines = 0


    # ========================================================
    # CHECK EVERY LABEL FILE
    # ========================================================

    for label_file in label_files:

        path = os.path.join(
            label_folder,
            label_file
        )


        with open(
                path,
                "r"
        ) as file:

            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]


        split_total_lines += len(lines)


        # ----------------------------------------------------
        # Find EXACT duplicate annotations
        # ----------------------------------------------------

        counts = Counter(lines)


        duplicates = {
            line: count
            for line, count in counts.items()
            if count > 1
        }


        unique_count = len(counts)

        split_unique_lines += unique_count


        # ----------------------------------------------------
        # Duplicates found
        # ----------------------------------------------------

        if duplicates:

            split_duplicate_files += 1
            total_duplicate_files += 1


            duplicate_count = sum(
                count - 1
                for count in duplicates.values()
            )


            split_duplicate_lines += duplicate_count
            total_duplicate_lines += duplicate_count


            print("\nDUPLICATES FOUND:")

            print(
                f"  {label_file}"
            )

            print(
                f"  Total annotations : "
                f"{len(lines)}"
            )

            print(
                f"  Unique annotations: "
                f"{unique_count}"
            )

            print(
                f"  Duplicate entries : "
                f"{duplicate_count}"
            )


            # ------------------------------------------------
            # Show duplicated classes
            # ------------------------------------------------

            for line, count in duplicates.items():

                parts = line.split()

                class_id = int(
                    parts[0]
                )

                class_name = CLASS_NAMES.get(
                    class_id,
                    "UNKNOWN"
                )


                print(
                    f"    {class_name}: "
                    f"appears {count} times"
                )


    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    print("\n----------------------------------------------")

    print(
        f"Label files                 : "
        f"{len(label_files)}"
    )

    print(
        f"Total annotation lines      : "
        f"{split_total_lines}"
    )

    print(
        f"Unique annotation lines     : "
        f"{split_unique_lines}"
    )

    print(
        f"Duplicate entries           : "
        f"{split_duplicate_lines}"
    )

    print(
        f"Files containing duplicates : "
        f"{split_duplicate_files}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("FINAL AUDIT")
print("==============================================")


print(
    f"\nFiles containing duplicates: "
    f"{total_duplicate_files}"
)

print(
    f"Duplicate annotation entries: "
    f"{total_duplicate_lines}"
)


if total_duplicate_lines == 0:

    print(
        "\nGOOD: No exact duplicate annotations found."
    )

else:

    print(
        "\nWARNING: The dataset contains "
        "duplicate annotations."
    )

    print(
        "We should investigate/clean these "
        "before retraining."
    )


print(
    "\n=============================================="
)

print(
    "AUDIT COMPLETE"
)

print(
    "=============================================="
)