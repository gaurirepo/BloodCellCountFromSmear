"""Canonical evaluation constants and count-error helpers.

Two protocols are kept strictly separate:

1. Scientific / COCO benchmark
   YOLO ``model.val(..., conf=0.001)`` → mAP@50, mAP@50-95, precision, recall.
   These numbers describe detector quality over the full PR curve.

2. Clinical / Streamlit operating point
   Infer at ``min(CLASS_THRESHOLDS)``, then accept a box only if its
   confidence meets the class-specific gate (RBC 0.60, WBC 0.40,
   Platelets 0.40). Count MAE / MAPE are computed at this gate only.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import csv
import json


PROJECT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# Canonical class mapping used by training, inference, and the app.
# Do not invert these IDs. TXL-PBC source labels are remapped below.
# ------------------------------------------------------------------

CANONICAL_CLASS_NAMES: Dict[int, str] = {
    0: "Platelets",
    1: "RBC",
    2: "WBC",
}

CLASS_IDS: Dict[str, int] = {
    name: class_id for class_id, name in CANONICAL_CLASS_NAMES.items()
}

DISPLAY_CLASS_ORDER: Tuple[str, ...] = ("WBC", "RBC", "Platelets")

CLASS_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "WBC": (0, 102, 255),       # Blue
    "RBC": (255, 60, 60),       # Red
    "Platelets": (255, 200, 0),  # Yellow
}

# TXL-PBC labels in this repo's audit scripts used a different ID order.
# Remap source IDs onto CANONICAL_CLASS_NAMES before any metric is computed.
TXL_PBC_SOURCE_NAMES: Dict[int, str] = {
    0: "WBC",
    1: "RBC",
    2: "Platelets",
}

TXL_PBC_ID_REMAP: Dict[int, int] = {
    0: 2,  # WBC        -> canonical WBC
    1: 1,  # RBC        -> canonical RBC
    2: 0,  # Platelets  -> canonical Platelets
}

# ------------------------------------------------------------------
# Protocol A — scientific COCO-style detector evaluation
# ------------------------------------------------------------------

COCO_VAL_CONF = 0.001
IMAGE_SIZE = 640
TEST_IMAGE_COUNT = 36
TEST_INSTANCE_COUNT = 471

SCIENTIFIC_METRICS = {
    "map50": 85.40,
    "map50_95": 60.07,
    "precision": 82.46,
    "per_class_map50": {
        "WBC": 96.90,
        "RBC": 85.40,
        "Platelets": 73.90,
    },
    "per_class_precision": {
        "WBC": 97.20,
        "RBC": 76.10,
        "Platelets": 74.20,
    },
}

BASELINE_METRICS = {
    "map50": 81.81,
    "map50_95": 56.47,
    "precision": 71.74,
    "per_class_map50": {
        "WBC": 96.90,
        "RBC": 83.40,
        "Platelets": 65.10,
    },
    "per_class_precision": {
        "WBC": 95.00,
        "RBC": 61.00,
        "Platelets": 59.20,
    },
}

# ------------------------------------------------------------------
# Protocol B — live inference / counting operating point
# ------------------------------------------------------------------

CLASS_THRESHOLDS: Dict[str, float] = {
    "RBC": 0.60,
    "WBC": 0.40,
    "Platelets": 0.40,
}

MIN_INFERENCE_CONF = min(CLASS_THRESHOLDS.values())

# Named examples from the audit; any image with GT count 0 is handled
# the same way (MAE is defined, MAPE is skipped for that class).
ZERO_GT_EXAMPLE_STEMS = (
    "BloodImage_00234",
    "BloodImage_00239",
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

COUNT_METRICS_JSON = (
    PROJECT_DIR / "runs" / "yolo26_evaluation" / "count_metrics.json"
)
COUNT_METRICS_CSV = (
    PROJECT_DIR / "runs" / "yolo26_evaluation" / "count_metrics.csv"
)

FINAL_MODEL_CANDIDATES = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "yolo26"
    / "full_training_from_corrected-2"
    / "weights"
    / "best.pt",
    PROJECT_DIR / "runs" / "yolo26" / "full_training" / "weights" / "best.pt",
)

TXL_PBC_CANDIDATES = (
    PROJECT_DIR / "TXL-PBC-CLEAN",
    PROJECT_DIR / "TXL-PBC",
    PROJECT_DIR / "external" / "TXL-PBC",
)


def empty_class_counts() -> Dict[str, int]:
    return {name: 0 for name in CANONICAL_CLASS_NAMES.values()}


def normalize_class_names(names: Mapping[Any, Any]) -> Dict[int, str]:
    return {int(class_id): str(class_name) for class_id, class_name in names.items()}


def assert_canonical_mapping(names: Mapping[Any, Any]) -> Dict[int, str]:
    """Raise if a YOLO model or data.yaml does not use the training taxonomy."""

    observed = normalize_class_names(names)
    if observed != CANONICAL_CLASS_NAMES:
        raise ValueError(
            "Class mapping mismatch.\n"
            f"Expected: {CANONICAL_CLASS_NAMES}\n"
            f"Found:    {observed}\n"
            "Inference scripts must keep "
            "{0: 'Platelets', 1: 'RBC', 2: 'WBC'}."
        )
    return observed


def resolve_dataset_root(project_dir: Path = PROJECT_DIR) -> Path:
    for name in ("DataSet", "Dataset"):
        candidate = project_dir / name
        if (candidate / "test" / "images").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find DataSet/test/images under {project_dir}"
    )


def resolve_final_model(project_dir: Path = PROJECT_DIR) -> Path:
    for candidate in (
        project_dir
        / "runs"
        / "detect"
        / "runs"
        / "yolo26"
        / "full_training_from_corrected-2"
        / "weights"
        / "best.pt",
        project_dir / "runs" / "yolo26" / "full_training" / "weights" / "best.pt",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Final YOLO weights not found. Expected best.pt under "
        "runs/detect/runs/yolo26/full_training_from_corrected-2/weights/"
    )


def list_images(images_dir: Path) -> List[Path]:
    images = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort()
    return images


def count_yolo_label_file(
    label_path: Path,
    id_remap: Optional[Mapping[int, int]] = None,
) -> Dict[str, int]:
    """Count class instances in a YOLO txt file.

    Missing or empty files are treated as zero annotations (no crash).
    Unknown class IDs are ignored rather than raising.
    """

    counts = empty_class_counts()
    if not label_path.is_file():
        return counts

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return counts

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if id_remap is not None:
            if class_id not in id_remap:
                continue
            class_id = id_remap[class_id]
        class_name = CANONICAL_CLASS_NAMES.get(class_id)
        if class_name is None:
            continue
        counts[class_name] += 1

    return counts


def label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def threshold_for_class(class_name: str) -> float:
    return CLASS_THRESHOLDS.get(class_name, MIN_INFERENCE_CONF)


def accept_detection(class_name: str, confidence: float) -> bool:
    return confidence >= threshold_for_class(class_name)


def count_accepted_boxes(
    class_ids: Sequence[int],
    confidences: Sequence[float],
    names: Mapping[int, str],
) -> Dict[str, int]:
    counts = empty_class_counts()
    normalized = normalize_class_names(names)
    for class_id, confidence in zip(class_ids, confidences):
        class_name = normalized.get(int(class_id))
        if class_name is None or class_name not in counts:
            continue
        if accept_detection(class_name, float(confidence)):
            counts[class_name] += 1
    return counts


def safe_mape(predicted: float, ground_truth: float) -> Optional[float]:
    """Absolute percentage error, or None when GT is 0 (undefined MAPE)."""

    if ground_truth == 0:
        return None
    return abs(predicted - ground_truth) / ground_truth * 100.0


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_count_errors(
    per_image: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """MAE over all images; MAPE only over images with GT > 0 for that class."""

    class_stats: Dict[str, Any] = {}

    for class_name in DISPLAY_CLASS_ORDER:
        abs_errors: List[float] = []
        percentage_errors: List[float] = []
        zero_gt_images: List[str] = []

        for row in per_image:
            gt = int(row["gt"][class_name])
            pred = int(row["pred"][class_name])
            abs_errors.append(abs(pred - gt))
            mape_value = safe_mape(pred, gt)
            if mape_value is None:
                zero_gt_images.append(str(row["image"]))
            else:
                percentage_errors.append(mape_value)

        class_stats[class_name] = {
            "mae": _mean(abs_errors),
            "mape": _mean(percentage_errors),
            "n_images": len(per_image),
            "n_mape": len(percentage_errors),
            "n_zero_gt": len(zero_gt_images),
            "zero_gt_images": zero_gt_images,
            "gt_total": sum(int(row["gt"][class_name]) for row in per_image),
            "pred_total": sum(int(row["pred"][class_name]) for row in per_image),
        }

    overall_abs: List[float] = []
    overall_pct: List[float] = []
    empty_images: List[str] = []

    for row in per_image:
        gt_total = sum(int(row["gt"][name]) for name in DISPLAY_CLASS_ORDER)
        pred_total = sum(int(row["pred"][name]) for name in DISPLAY_CLASS_ORDER)
        overall_abs.append(abs(pred_total - gt_total))
        mape_value = safe_mape(pred_total, gt_total)
        if mape_value is None:
            empty_images.append(str(row["image"]))
        else:
            overall_pct.append(mape_value)

    return {
        "n_images": len(per_image),
        "classes": class_stats,
        "overall": {
            "mae": _mean(overall_abs),
            "mape": _mean(overall_pct),
            "n_mape": len(overall_pct),
            "n_zero_gt": len(empty_images),
            "zero_gt_images": empty_images,
        },
        "operating_point": dict(CLASS_THRESHOLDS),
        "scientific_protocol": {
            "conf": COCO_VAL_CONF,
            "metric": "COCO mAP@50 / mAP@50-95 / precision",
        },
    }


def write_count_metrics(
    summary: Mapping[str, Any],
    per_image: Sequence[Mapping[str, Any]],
    json_path: Path = COUNT_METRICS_JSON,
    csv_path: Path = COUNT_METRICS_CSV,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **summary,
        "per_image": list(per_image),
    }
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Image",
                "Actual WBC",
                "Predicted WBC",
                "Actual RBC",
                "Predicted RBC",
                "Actual Platelets",
                "Predicted Platelets",
            ]
        )
        for row in per_image:
            writer.writerow(
                [
                    row["image"],
                    row["gt"]["WBC"],
                    row["pred"]["WBC"],
                    row["gt"]["RBC"],
                    row["pred"]["RBC"],
                    row["gt"]["Platelets"],
                    row["pred"]["Platelets"],
                ]
            )


def load_count_metrics(
    json_path: Path = COUNT_METRICS_JSON,
) -> Optional[Dict[str, Any]]:
    if not json_path.is_file():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def format_metric(value: Optional[float], suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def find_txl_pbc_root(project_dir: Path = PROJECT_DIR) -> Optional[Path]:
    for candidate in (
        project_dir / "TXL-PBC-CLEAN",
        project_dir / "TXL-PBC",
        project_dir / "external" / "TXL-PBC",
    ):
        if candidate.is_dir():
            return candidate
    return None


def remap_txl_label_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        destination.write_text("", encoding="utf-8")
        return

    remapped: List[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id not in TXL_PBC_ID_REMAP:
            continue
        parts[0] = str(TXL_PBC_ID_REMAP[class_id])
        remapped.append(" ".join(parts))
    destination.write_text("\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")
