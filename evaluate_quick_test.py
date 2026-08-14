"""Test-set evaluation with two strictly separated protocols.

Protocol A — Scientific / COCO benchmark
    model.val(split='test', conf=0.001) → mAP@50, precision, recall.

Protocol B — Clinical operating point (same gates as app.py)
    Infer at min class threshold, then accept:
        RBC ≥ 0.60, WBC ≥ 0.40, Platelets ≥ 0.40
    Count MAE / MAPE vs manual YOLO labels on all 36 test images.

Images with zero ground-truth boxes (including named examples such as
BloodImage_00234 / BloodImage_00239) contribute to MAE but are excluded
from MAPE so there is no division by zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval_config import (
    CANONICAL_CLASS_NAMES,
    CLASS_THRESHOLDS,
    COCO_VAL_CONF,
    COUNT_METRICS_CSV,
    COUNT_METRICS_JSON,
    DISPLAY_CLASS_ORDER,
    IMAGE_SIZE,
    MIN_INFERENCE_CONF,
    ZERO_GT_EXAMPLE_STEMS,
    assert_canonical_mapping,
    count_accepted_boxes,
    count_yolo_label_file,
    format_metric,
    label_path_for_image,
    list_images,
    resolve_dataset_root,
    resolve_final_model,
    summarize_count_errors,
    write_count_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "data.yaml"
OUTPUT_DIR = PROJECT_DIR / "runs" / "yolo26_evaluation" / "quick_test"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COCO mAP (conf=0.001) plus operating-point count MAE/MAPE."
    )
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="Skip YOLO model.val COCO benchmark (still runs count MAE/MAPE).",
    )
    parser.add_argument(
        "--skip-counts",
        action="store_true",
        help="Skip operating-point counting and MAE/MAPE.",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not write annotated prediction images.",
    )
    return parser.parse_args(argv)


def _print_mapping(names: Dict[int, str]) -> None:
    print("Canonical class mapping (must not be remapped at inference):")
    for class_id in sorted(names):
        print(f"  {class_id} -> {names[class_id]}")


def _print_protocol_banner() -> None:
    print("=" * 78)
    print("EVALUATION PROTOCOLS (DO NOT CONFLATE)")
    print("=" * 78)
    print()
    print("A. Scientific benchmark  (COCO)")
    print(f"   conf = {COCO_VAL_CONF}  →  mAP@50 / mAP@50-95 / precision / recall")
    print("   This is NOT the Streamlit counting threshold.")
    print()
    print("B. Live inference / counting  (clinical operating point)")
    print(f"   infer conf = {MIN_INFERENCE_CONF:.2f}, then class gates:")
    for class_name in DISPLAY_CLASS_ORDER:
        print(f"     {class_name:10s}  ≥ {CLASS_THRESHOLDS[class_name]:.2f}")
    print("   Metrics: count MAE and MAPE vs manual labels.")
    print()


def run_coco_benchmark(model: Any, data_yaml: Path) -> Any:
    print("=" * 78)
    print("PROTOCOL A — SCIENTIFIC COCO BENCHMARK")
    print("=" * 78)
    print(f"Validation confidence: {COCO_VAL_CONF}")
    print(f"Image size:            {IMAGE_SIZE}")
    print()

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        conf=COCO_VAL_CONF,
        plots=True,
        verbose=True,
        project=str(OUTPUT_DIR.parent),
        name="accuracy",
        exist_ok=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print()
    print("COCO TEST RESULTS  (conf=0.001, not the app operating point)")
    print("-" * 78)
    print(f"Precision : {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"F1        : {f1:.4f}  ({f1 * 100:.2f}%)")
    print(f"mAP50     : {map50:.4f}  ({map50 * 100:.2f}%)")
    print(f"mAP50-95  : {map5095:.4f}  ({map5095 * 100:.2f}%)")
    print()
    print("| Class | Precision | Recall | mAP50 | mAP50-95 |")
    print("|---|---:|---:|---:|---:|")

    for class_id, class_name in CANONICAL_CLASS_NAMES.items():
        try:
            p = float(metrics.box.p[class_id])
            r = float(metrics.box.r[class_id])
            ap50 = float(metrics.box.ap50[class_id])
            ap = float(metrics.box.ap[class_id])
        except Exception as exc:  # noqa: BLE001 — Ultralytics metric vectors vary
            print(f"| {class_name} | n/a | n/a | n/a | n/a |  ({exc})")
            continue
        print(
            f"| {class_name} "
            f"| {p * 100:.2f}% "
            f"| {r * 100:.2f}% "
            f"| {ap50 * 100:.2f}% "
            f"| {ap * 100:.2f}% |"
        )

    print(
        f"| Overall "
        f"| {precision * 100:.2f}% "
        f"| {recall * 100:.2f}% "
        f"| {map50 * 100:.2f}% "
        f"| {map5095 * 100:.2f}% |"
    )
    print()
    return metrics


def run_operating_point_counts(
    model: Any,
    images: List[Path],
    labels_dir: Path,
    save_predictions: bool,
) -> Dict[str, Any]:
    print("=" * 78)
    print("PROTOCOL B — CLINICAL OPERATING POINT  (count MAE / MAPE)")
    print("=" * 78)
    print(f"YOLO infer conf: {MIN_INFERENCE_CONF:.2f}")
    print("Class acceptance gates:")
    for class_name in DISPLAY_CLASS_ORDER:
        print(f"  {class_name:10s}  {CLASS_THRESHOLDS[class_name]:.2f}")
    print()
    print(
        "Zero-GT images (empty labels or named examples "
        f"{', '.join(ZERO_GT_EXAMPLE_STEMS)}) are included in MAE "
        "and excluded from MAPE."
    )
    print()

    names = assert_canonical_mapping(model.names)
    per_image = []
    prediction_output = OUTPUT_DIR / "predictions"

    for index, image_path in enumerate(images, start=1):
        predict_kwargs = {
            "source": str(image_path),
            "conf": MIN_INFERENCE_CONF,
            "imgsz": IMAGE_SIZE,
            "save": save_predictions,
            "verbose": False,
        }
        if save_predictions:
            predict_kwargs["project"] = str(prediction_output.parent)
            predict_kwargs["name"] = prediction_output.name
            predict_kwargs["exist_ok"] = True

        results = model.predict(**predict_kwargs)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = [int(value) for value in result.boxes.cls.tolist()]
            confidences = [float(value) for value in result.boxes.conf.tolist()]
        else:
            class_ids = []
            confidences = []

        pred = count_accepted_boxes(class_ids, confidences, names)
        gt = count_yolo_label_file(label_path_for_image(image_path, labels_dir))
        gt_total = sum(gt.values())

        per_image.append(
            {
                "image": image_path.name,
                "stem": image_path.stem,
                "gt": gt,
                "pred": pred,
                "zero_gt": gt_total == 0,
            }
        )

        marker = "  [zero-GT]" if gt_total == 0 else ""
        print(
            f"{index:3}/{len(images)}  {image_path.name}{marker}\n"
            f"       GT   WBC={gt['WBC']:2d}  RBC={gt['RBC']:3d}  "
            f"Platelets={gt['Platelets']:2d}\n"
            f"       Pred WBC={pred['WBC']:2d}  RBC={pred['RBC']:3d}  "
            f"Platelets={pred['Platelets']:2d}"
        )

    summary = summarize_count_errors(per_image)
    write_count_metrics(summary, per_image)

    print()
    print("COUNT ERROR  (operating point, not COCO mAP)")
    print("-" * 78)
    print(
        f"{'Class':<12}{'MAE':>10}{'MAPE':>12}{'n MAPE':>10}"
        f"{'zero-GT':>10}{'GT':>8}{'Pred':>8}"
    )
    for class_name in DISPLAY_CLASS_ORDER:
        stats = summary["classes"][class_name]
        print(
            f"{class_name:<12}"
            f"{format_metric(stats['mae']):>10}"
            f"{format_metric(stats['mape'], '%'):>12}"
            f"{stats['n_mape']:>10}"
            f"{stats['n_zero_gt']:>10}"
            f"{stats['gt_total']:>8}"
            f"{stats['pred_total']:>8}"
        )

    overall = summary["overall"]
    print("-" * 78)
    print(
        f"{'Overall':<12}"
        f"{format_metric(overall['mae']):>10}"
        f"{format_metric(overall['mape'], '%'):>12}"
        f"{overall['n_mape']:>10}"
        f"{overall['n_zero_gt']:>10}"
    )
    print()
    print(f"Wrote {COUNT_METRICS_JSON}")
    print(f"Wrote {COUNT_METRICS_CSV}")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("=" * 78)
    print("YOLO26 TEST EVALUATION — mAP vs OPERATING-POINT COUNTS")
    print("=" * 78)
    print()

    try:
        model_path = resolve_final_model(PROJECT_DIR)
        dataset_root = resolve_dataset_root(PROJECT_DIR)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    test_images_dir = dataset_root / "test" / "images"
    test_labels_dir = dataset_root / "test" / "labels"

    print(f"Model:      {model_path}")
    print(f"data.yaml:  {DATA_YAML}")
    print(f"Test images:{test_images_dir}")
    print()
    _print_mapping(CANONICAL_CLASS_NAMES)
    print()
    _print_protocol_banner()

    if not DATA_YAML.exists():
        print(f"ERROR: data.yaml not found: {DATA_YAML}")
        return 1

    images = list_images(test_images_dir)
    print(f"Test images found: {len(images)}")
    if not images:
        print("ERROR: no test images.")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    try:
        assert_canonical_mapping(model.names)
        print("Class mapping VERIFIED against canonical taxonomy.")
    except ValueError as exc:
        print(exc)
        return 1
    print()

    if not args.skip_coco:
        run_coco_benchmark(model, DATA_YAML)
    else:
        print("Skipping Protocol A (--skip-coco).")
        print()

    if not args.skip_counts:
        run_operating_point_counts(
            model=model,
            images=images,
            labels_dir=test_labels_dir,
            save_predictions=not args.no_save_predictions,
        )
    else:
        print("Skipping Protocol B (--skip-counts).")

    print("=" * 78)
    print("EVALUATION COMPLETE")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
