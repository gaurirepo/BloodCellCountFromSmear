from pathlib import Path
from collections import Counter

# ============================================================
# DATASET LABEL AUDIT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "Dataset"

SETS = ["train", "valid", "test"]

CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC"
}

print("=" * 75)
print("BLOOD CELL DATASET LABEL AUDIT")
print("=" * 75)

overall = Counter()

for split in SETS:

    labels_dir = DATASET_DIR / split / "labels"
    images_dir = DATASET_DIR / split / "images"

    print("\n" + "=" * 75)
    print(f"{split.upper()} DATASET")
    print("=" * 75)

    if not labels_dir.exists():
        print(f"ERROR: {labels_dir} does not exist")
        continue

    label_files = sorted(labels_dir.glob("*.txt"))

    print(f"Images directory : {images_dir}")
    print(f"Labels directory : {labels_dir}")
    print(f"Label files      : {len(label_files)}")

    total_counts = Counter()

    zero_rbc = []
    zero_platelets = []
    zero_wbc = []

    suspicious = []

    for label_file in label_files:

        counts = Counter()

        try:
            with open(label_file, "r") as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])

                    if class_id in CLASS_NAMES:
                        counts[class_id] += 1
                        total_counts[class_id] += 1
                        overall[class_id] += 1

        except Exception as e:
            print(f"ERROR reading {label_file.name}: {e}")
            continue

        rbc = counts[1]
        platelet = counts[0]
        wbc = counts[2]

        # ----------------------------------------------------
        # Find images with ZERO RBC labels
        # ----------------------------------------------------

        if rbc == 0:
            zero_rbc.append(label_file.name)

        if platelet == 0:
            zero_platelets.append(label_file.name)

        if wbc == 0:
            zero_wbc.append(label_file.name)

        # ----------------------------------------------------
        # Particularly suspicious:
        # WBC exists but ZERO RBC
        # ----------------------------------------------------

        if wbc > 0 and rbc == 0:
            suspicious.append(
                (label_file.name, platelet, rbc, wbc)
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nTOTAL LABELED OBJECTS")
    print("-" * 50)

    for class_id in [0, 1, 2]:

        print(
            f"{CLASS_NAMES[class_id]:12s}: "
            f"{total_counts[class_id]}"
        )

    print("\nIMAGES WITH ZERO LABELS FOR CLASS")
    print("-" * 50)

    print(f"Platelets : {len(zero_platelets)}")
    print(f"RBC       : {len(zero_rbc)}")
    print(f"WBC       : {len(zero_wbc)}")

    # ========================================================
    # SUSPICIOUS WBC-ONLY IMAGES
    # ========================================================

    print("\n" + "-" * 75)
    print("POTENTIALLY SUSPICIOUS: WBC PRESENT BUT ZERO RBC")
    print("-" * 75)

    print(f"Count: {len(suspicious)}")

    if suspicious:

        for filename, platelet, rbc, wbc in suspicious[:100]:

            print(
                f"{filename:65s} "
                f"Platelets={platelet:2d} "
                f"RBC={rbc:2d} "
                f"WBC={wbc:2d}"
            )

        if len(suspicious) > 100:
            print(
                f"\n... and {len(suspicious) - 100} more"
            )

    # ========================================================
    # RBC DISTRIBUTION
    # ========================================================

    print("\nRBC LABEL COUNT DISTRIBUTION")
    print("-" * 50)

    distribution = Counter()

    for label_file in label_files:

        rbc_count = 0

        try:
            with open(label_file, "r") as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) >= 5 and int(parts[0]) == 1:
                        rbc_count += 1

        except:
            continue

        if rbc_count == 0:
            distribution["0 RBC"] += 1
        elif rbc_count <= 5:
            distribution["1-5 RBC"] += 1
        elif rbc_count <= 10:
            distribution["6-10 RBC"] += 1
        elif rbc_count <= 20:
            distribution["11-20 RBC"] += 1
        else:
            distribution["21+ RBC"] += 1

    for category in [
        "0 RBC",
        "1-5 RBC",
        "6-10 RBC",
        "11-20 RBC",
        "21+ RBC"
    ]:

        print(
            f"{category:12s}: "
            f"{distribution[category]}"
        )


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("OVERALL DATASET SUMMARY")
print("=" * 75)

for class_id in [0, 1, 2]:

    print(
        f"{CLASS_NAMES[class_id]:12s}: "
        f"{overall[class_id]} labeled objects"
    )

print("\n" + "=" * 75)
print("AUDIT COMPLETE")
print("=" * 75)

print("""
IMPORTANT:

This script only audits the LABEL FILES.

A label count of RBC=0 does NOT automatically prove
that RBCs are missing from the annotation.

Those images should be visually inspected.

The most important group to inspect is:

    WBC > 0
    RBC = 0

because these images may contain visible RBCs that
were never annotated.
""")