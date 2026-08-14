"""External validation on TXL-PBC with a frozen canonical class map.

This repository trains and infers with:

    {0: 'Platelets', 1: 'RBC', 2: 'WBC'}

TXL-PBC dumps used by the audit scripts in this repo used a different
order ({0: 'WBC', 1: 'RBC', 2: 'Platelets'}). Labels are remapped onto
the canonical IDs before any metric is computed. Weights and Streamlit
thresholds are not tuned on this set.

If TXL-PBC is not present, the script validates the remapping table and
exits 0 so CI / syntax checks still pass.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from eval_config import (
    CANONICAL_CLASS_NAMES,
    CLASS_THRESHOLDS,
    COCO_VAL_CONF,
    DISPLAY_CLASS_ORDER,
    IMAGE_SIZE,
    MIN_INFERENCE_CONF,
    TXL_PBC_ID_REMAP,
    TXL_PBC_SOURCE_NAMES,
    assert_canonical_mapping,
    count_accepted_boxes,
    count_yolo_label_file,
    find_txl_pbc_root,
    format_metric,
    label_path_for_image,
    list_images,
    remap_txl_label_file,
    resolve_final_model,
    summarize_count_errors,
    write_count_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parent
EXTERNAL_OUTPUT = PROJECT_DIR / "runs" / "yolo26_evaluation" / "txl_pbc"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TXL-PBC external validation with canonical class remapping."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="TXL-PBC root (default: TXL-PBC-CLEAN, then TXL-PBC).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to evaluate (test, val, or train).",
    )
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="Skip mAP val; only compute operating-point MAE/MAPE.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Remap labels and write data yaml, then exit.",
    )
    return parser.parse_args(argv)


def _split_dirs(root: Path, split: str) -> Optional[tuple]:
    candidates = [
        (root / split / "images", root / split / "labels"),
        (root / "images" / split, root / "labels" / split),
        (root / "images" / split.capitalize(), root / "labels" / split.capitalize()),
    ]
    for images_dir, labels_dir in candidates:
        if images_dir.is_dir():
            return images_dir, labels_dir
    return None


def prepare_canonical_split(
    source_root: Path,
    split: str,
    dest_root: Path,
) -> Path:
    """Copy images and remap TXL label IDs onto the canonical taxonomy."""

    found = _split_dirs(source_root, split)
    if found is None:
        raise FileNotFoundError(
            f"Could not find {split} images under {source_root}. "
            "Expected {split}/images or images/{split}."
        )

    src_images, src_labels = found
    dest_images = dest_root / "images" / split
    dest_labels = dest_root / "labels" / split
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    images = list_images(src_images)
    if not images:
        raise FileNotFoundError(f"No images in {src_images}")

    for image_path in images:
        target_image = dest_images / image_path.name
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
        remap_txl_label_file(
            label_path_for_image(image_path, src_labels),
            dest_labels / f"{image_path.stem}.txt",
        )

    yaml_path = dest_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dest_root.resolve()}",
                "train: images/train",
                f"val: images/{split}",
                f"test: images/{split}",
                "nc: 3",
                "names:",
                "  0: Platelets",
                "  1: RBC",
                "  2: WBC",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def print_mapping_contract() -> None:
    print("Canonical inference mapping (frozen):")
    for class_id, name in CANONICAL_CLASS_NAMES.items():
        print(f"  {class_id} -> {name}")
    print()
    print("TXL-PBC source mapping (audit-script order) remapped as:")
    for source_id, source_name in TXL_PBC_SOURCE_NAMES.items():
        dest_id = TXL_PBC_ID_REMAP[source_id]
        dest_name = CANONICAL_CLASS_NAMES[dest_id]
        print(f"  TXL {source_id} {source_name:10s} -> canonical {dest_id} {dest_name}")
    print()
    print("Operating-point gates (not used for mAP):")
    for class_name in DISPLAY_CLASS_ORDER:
        print(f"  {class_name:10s} ≥ {CLASS_THRESHOLDS[class_name]:.2f}")
    print(f"COCO val conf: {COCO_VAL_CONF}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print("=" * 78)
    print("TXL-PBC EXTERNAL VALIDATION")
    print("=" * 78)
    print()
    print_mapping_contract()

    source_root = args.data_root or find_txl_pbc_root(PROJECT_DIR)
    if source_root is None:
        print(
            "TXL-PBC images are not in this checkout.\n"
            "Place the dataset at TXL-PBC/ or TXL-PBC-CLEAN/ "
            "(test/images + test/labels, or images/test + labels/test),\n"
            "then re-run:\n"
            "  python validate_txl_pbc.py --split test\n"
        )
        print("Remap table verified. Nothing to evaluate yet.")
        return 0

    print(f"TXL-PBC root: {source_root}")
    dest_root = EXTERNAL_OUTPUT / "canonical_dataset"
    try:
        yaml_path = prepare_canonical_split(source_root, args.split, dest_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Canonical data yaml: {yaml_path}")
    if args.prepare_only:
        print("Prepare-only: remapped labels written, skipping inference.")
        return 0

    try:
        model_path = resolve_final_model(PROJECT_DIR)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    try:
        assert_canonical_mapping(model.names)
    except ValueError as exc:
        print(exc)
        return 1

    images_dir = dest_root / "images" / args.split
    labels_dir = dest_root / "labels" / args.split
    images = list_images(images_dir)
    print(f"Evaluating {len(images)} {args.split} images (weights frozen).")
    print()

    if not args.skip_coco:
        print("Protocol A — COCO mAP at conf=0.001 (frozen thresholds unused).")
        model.val(
            data=str(yaml_path),
            split="test",
            imgsz=IMAGE_SIZE,
            conf=COCO_VAL_CONF,
            plots=True,
            verbose=True,
            project=str(EXTERNAL_OUTPUT),
            name="coco",
            exist_ok=True,
        )
        print()

    print("Protocol B — operating-point count MAE / MAPE.")
    per_image = []
    for image_path in images:
        results = model.predict(
            source=str(image_path),
            conf=MIN_INFERENCE_CONF,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = [int(value) for value in result.boxes.cls.tolist()]
            confidences = [float(value) for value in result.boxes.conf.tolist()]
        else:
            class_ids = []
            confidences = []
        pred = count_accepted_boxes(class_ids, confidences, model.names)
        gt = count_yolo_label_file(label_path_for_image(image_path, labels_dir))
        per_image.append(
            {
                "image": image_path.name,
                "gt": gt,
                "pred": pred,
            }
        )

    summary = summarize_count_errors(per_image)
    json_path = EXTERNAL_OUTPUT / "count_metrics.json"
    csv_path = EXTERNAL_OUTPUT / "count_metrics.csv"
    write_count_metrics(summary, per_image, json_path=json_path, csv_path=csv_path)

    print(
        f"{'Class':<12}{'MAE':>10}{'MAPE':>12}{'n MAPE':>10}{'zero-GT':>10}"
    )
    for class_name in DISPLAY_CLASS_ORDER:
        stats = summary["classes"][class_name]
        print(
            f"{class_name:<12}"
            f"{format_metric(stats['mae']):>10}"
            f"{format_metric(stats['mape'], '%'):>12}"
            f"{stats['n_mape']:>10}"
            f"{stats['n_zero_gt']:>10}"
        )
    print(f"\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
