from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

print("=" * 70)
print("TEST DATASET LABEL AUDIT")
print("=" * 70)

# --------------------------------------------------
# PROJECT LOCATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_DIR / "Dataset/test/images"
LABEL_DIR = PROJECT_DIR / "Dataset/test/labels"

OUTPUT_DIR = PROJECT_DIR / "runs/yolo26_evaluation/test_label_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# CLASS MAPPING
# --------------------------------------------------

CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC"
}

# --------------------------------------------------
# CHECK DIRECTORIES
# --------------------------------------------------

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found:\n{IMAGE_DIR}")

if not LABEL_DIR.exists():
    raise FileNotFoundError(f"Label directory not found:\n{LABEL_DIR}")

print(f"\nImages : {IMAGE_DIR}")
print(f"Labels : {LABEL_DIR}")
print(f"Output : {OUTPUT_DIR}")

# --------------------------------------------------
# GET TEST IMAGES
# --------------------------------------------------

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

print(f"\nNumber of test images: {len(image_files)}")

if len(image_files) == 0:
    raise RuntimeError("No test images found.")

# --------------------------------------------------
# FONT
# --------------------------------------------------

try:
    font = ImageFont.truetype(
        "/System/Library/Fonts/Helvetica.ttc",
        14
    )
except:
    font = ImageFont.load_default()

# --------------------------------------------------
# PROCESS EACH IMAGE
# --------------------------------------------------

total_counts = Counter()

missing_labels = []
invalid_labels = []

print("\n" + "=" * 70)
print("GROUND-TRUTH LABEL SUMMARY")
print("=" * 70)

for index, image_path in enumerate(image_files, start=1):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    print(f"\n{index:02d}/{len(image_files)}  {image_path.name}")

    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    draw = ImageDraw.Draw(image)

    # --------------------------------------------------
    # CHECK LABEL EXISTS
    # --------------------------------------------------

    if not label_path.exists():

        print("  !!! LABEL FILE MISSING !!!")

        missing_labels.append(image_path.name)

        output_path = OUTPUT_DIR / image_path.name
        image.save(output_path)

        continue

    # --------------------------------------------------
    # READ LABELS
    # --------------------------------------------------

    counts = Counter()

    with open(label_path, "r") as f:

        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        if len(parts) != 5:

            print(
                f"  !!! INVALID LABEL FORMAT "
                f"(line {line_number}): {line}"
            )

            invalid_labels.append(
                f"{image_path.name}: line {line_number}"
            )

            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

        except ValueError:

            print(
                f"  !!! INVALID NUMERIC LABEL "
                f"(line {line_number}): {line}"
            )

            invalid_labels.append(
                f"{image_path.name}: line {line_number}"
            )

            continue

        # --------------------------------------------------
        # CHECK CLASS
        # --------------------------------------------------

        if class_id not in CLASS_NAMES:

            print(
                f"  !!! UNKNOWN CLASS {class_id} "
                f"(line {line_number})"
            )

            invalid_labels.append(
                f"{image_path.name}: line {line_number}"
            )

            continue

        class_name = CLASS_NAMES[class_id]

        counts[class_name] += 1
        total_counts[class_name] += 1

        # --------------------------------------------------
        # CONVERT YOLO NORMALIZED COORDINATES
        # TO PIXEL COORDINATES
        # --------------------------------------------------

        x_center_px = x_center * width
        y_center_px = y_center * height

        box_width_px = box_width * width
        box_height_px = box_height * height

        x1 = x_center_px - box_width_px / 2
        y1 = y_center_px - box_height_px / 2

        x2 = x_center_px + box_width_px / 2
        y2 = y_center_px + box_height_px / 2

        # --------------------------------------------------
        # CLAMP TO IMAGE
        # --------------------------------------------------

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))

        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        # --------------------------------------------------
        # DRAW BOX
        # --------------------------------------------------

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=2
        )

        # --------------------------------------------------
        # DRAW LABEL
        # --------------------------------------------------

        text = f"{class_name}"

        bbox = draw.textbbox(
            (x1, y1),
            text,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_y = max(0, y1 - text_height - 4)

        draw.rectangle(
            [
                x1,
                text_y,
                x1 + text_width + 6,
                text_y + text_height + 4
            ],
            fill="red"
        )

        draw.text(
            (x1 + 3, text_y + 2),
            text,
            fill="white",
            font=font
        )

    # --------------------------------------------------
    # PRINT COUNTS
    # --------------------------------------------------

    print(
        f"  Platelets: {counts['Platelets']:2d} | "
        f"RBC: {counts['RBC']:2d} | "
        f"WBC: {counts['WBC']:2d} | "
        f"Total: {sum(counts.values()):2d}"
    )

    # --------------------------------------------------
    # SAVE ANNOTATED IMAGE
    # --------------------------------------------------

    output_path = OUTPUT_DIR / image_path.name

    image.save(output_path)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nTOTAL GROUND-TRUTH INSTANCES")
print("-" * 70)

print(f"Platelets : {total_counts['Platelets']}")
print(f"RBC       : {total_counts['RBC']}")
print(f"WBC       : {total_counts['WBC']}")
print(f"TOTAL     : {sum(total_counts.values())}")

# --------------------------------------------------
# PROBLEMS
# --------------------------------------------------

print("\n" + "=" * 70)
print("LABEL FILE CHECK")
print("=" * 70)

if missing_labels:

    print("\nMissing label files:")

    for name in missing_labels:
        print(f"  - {name}")

else:

    print("\nNo missing label files.")

if invalid_labels:

    print("\nInvalid labels:")

    for item in invalid_labels:
        print(f"  - {item}")

else:

    print("\nNo invalid label formats detected.")

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\n" + "=" * 70)
print("ANNOTATED TEST IMAGES")
print("=" * 70)

print(f"\n{OUTPUT_DIR}")

print(
    "\nOpen the folder above and visually inspect "
    "the red boxes against the actual cells."
)

print("\nFor each image check:")
print("  1. Is every visible RBC labelled?")
print("  2. Is every RBC box actually around an RBC?")
print("  3. Are platelets labelled as Platelets?")
print("  4. Is the WBC labelled as WBC?")
print("  5. Are there missing boxes?")
print("  6. Are there duplicate/overlapping boxes?")