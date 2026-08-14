from eval_config import (
    CANONICAL_CLASS_NAMES,
    CLASS_THRESHOLDS,
    DISPLAY_CLASS_ORDER,
    IMAGE_SIZE,
    MIN_INFERENCE_CONF,
    PROJECT_DIR,
    assert_canonical_mapping,
    count_accepted_boxes,
    empty_class_counts,
    list_images,
    resolve_dataset_root,
    resolve_final_model,
)


def main() -> None:
    from ultralytics import YOLO

    model_path = resolve_final_model(PROJECT_DIR)
    test_images = resolve_dataset_root(PROJECT_DIR) / "test" / "images"
    output_dir = PROJECT_DIR / "runs" / "yolo26_predictions" / "full_training"

    print("=" * 60)
    print("YOLO26 FULL TRAINED MODEL - BLOOD CELL PREDICTION")
    print("=" * 60)
    print()
    print("Model:")
    print(model_path)
    print()
    print("Test images:")
    print(test_images)
    print()
    print("Canonical class mapping:")
    print(CANONICAL_CLASS_NAMES)
    print()
    print("Live counting gates (not COCO mAP):")
    for class_name in DISPLAY_CLASS_ORDER:
        print(f"  {class_name}: {CLASS_THRESHOLDS[class_name]:.2f}")

    if not model_path.exists():
        raise FileNotFoundError(f"\nERROR: best.pt not found!\n{model_path}")
    if not test_images.exists():
        raise FileNotFoundError(
            f"\nERROR: Test image folder not found!\n{test_images}"
        )

    print("\nLoading model...")
    model = YOLO(str(model_path))
    print("Model loaded successfully!")
    print()
    print("Model classes:")
    print(model.names)

    print()
    print("Checking class mapping...")
    assert_canonical_mapping(model.names)
    print("Class mapping verified successfully!")

    images = list_images(test_images)
    print()
    print("Number of test images:", len(images))
    if not images:
        raise ValueError("No test images found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("Output directory:")
    print(output_dir)

    print()
    print("=" * 60)
    print("RUNNING PREDICTIONS")
    print("=" * 60)

    results = model.predict(
        source=str(test_images),
        conf=MIN_INFERENCE_CONF,
        imgsz=IMAGE_SIZE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=True,
    )

    totals = empty_class_counts()

    print()
    print("=" * 60)
    print("PREDICTION SUMMARY  (class-specific operating point)")
    print("=" * 60)

    for index, result in enumerate(results):
        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = [int(value) for value in result.boxes.cls.tolist()]
            confidences = [float(value) for value in result.boxes.conf.tolist()]
        else:
            class_ids = []
            confidences = []

        counts = count_accepted_boxes(class_ids, confidences, model.names)
        for class_name, value in counts.items():
            totals[class_name] += value

        print(
            f"{index + 1:3}/{len(results)}  "
            f"WBC={counts['WBC']:2}  "
            f"RBC={counts['RBC']:3}  "
            f"Platelets={counts['Platelets']:2}"
        )

    print()
    print("=" * 60)
    print("TOTAL PREDICTIONS")
    print("=" * 60)
    print(f"WBC        : {totals['WBC']}")
    print(f"RBC        : {totals['RBC']}")
    print(f"Platelets  : {totals['Platelets']}")
    print()
    print("Annotated images:")
    print(output_dir)
    print()
    print("Prediction labels:")
    print(output_dir / "labels")
    print()
    print("YOLO infer confidence:")
    print(MIN_INFERENCE_CONF)
    print()
    print("Done!")


if __name__ == "__main__":
    main()
