import os

# ============================================================
# IMAGE WE ARE INVESTIGATING
# ============================================================

image_name = "1299047edeb15e2581d9680aaa667d5d.png"

label_name = os.path.splitext(image_name)[0] + ".txt"

label_path = os.path.join(
    "DataSet_TXL_PBC",
    "labels",
    "test",
    label_name
)


# ============================================================
# READ LABEL
# ============================================================

print("\n========================================")
print("GROUND TRUTH LABEL CHECK")
print("========================================")

print("\nImage:")
print(image_name)

print("\nLabel file:")
print(label_path)


if not os.path.exists(label_path):

    print("\nERROR: Label file not found!")

else:

    counts = {
        0: 0,
        1: 0,
        2: 0
    }


    print("\nAnnotations:\n")


    with open(
            label_path,
            "r"
    ) as file:

        lines = file.readlines()


    for number, line in enumerate(
            lines,
            start=1
    ):

        parts = line.strip().split()

        if len(parts) != 5:

            print(
                "BAD LINE:",
                number,
                line
            )

            continue


        class_id = int(parts[0])

        x = float(parts[1])
        y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])


        counts[class_id] += 1


        class_name = {
            0: "WBC",
            1: "RBC",
            2: "Platelets"
        }.get(
            class_id,
            "UNKNOWN"
        )


        print(
            f"{number:2d}. "
            f"{class_name:<12} "
            f"x={x:.3f} "
            f"y={y:.3f} "
            f"w={width:.3f} "
            f"h={height:.3f}"
        )


    print("\n========================================")
    print("COUNTS IN DATASET LABEL")
    print("========================================")

    print(
        "\nWBC       :",
        counts[0]
    )

    print(
        "RBC       :",
        counts[1]
    )

    print(
        "Platelets :",
        counts[2]
    )

    print(
        "\nTotal     :",
        sum(counts.values())
    )