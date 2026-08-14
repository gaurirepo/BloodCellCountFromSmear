"""SmearDx — clinical Streamlit interface for YOLO blood-smear detection.

Inference protocol is unchanged:
  - YOLO is queried at min(class gates)
  - each box is accepted only if it meets its class-specific gate
  - COCO mAP@50 (conf=0.001) is never mixed with live counting
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from eval_config import (
    BASELINE_METRICS,
    CANONICAL_CLASS_NAMES,
    CLASS_THRESHOLDS,
    COCO_VAL_CONF,
    DISPLAY_CLASS_ORDER,
    IMAGE_SIZE,
    PROJECT_DIR,
    SCIENTIFIC_METRICS,
    TEST_IMAGE_COUNT,
    TEST_INSTANCE_COUNT,
    empty_class_counts,
    format_metric,
    load_count_metrics,
    resolve_final_model,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ------------------------------------------------------------------
# Visual tokens (clinical dark)
# ------------------------------------------------------------------

UI_BOX_COLORS: Dict[str, Tuple[int, int, int]] = {
    "RBC": (239, 68, 68),
    "WBC": (59, 130, 246),
    "Platelets": (245, 158, 11),
}

UI_HEX = {
    "RBC": "#EF4444",
    "WBC": "#3B82F6",
    "Platelets": "#F59E0B",
    "accent": "#3B82F6",
}

DEFAULT_IOU = 0.70


try:
    MODEL_PATH = resolve_final_model(PROJECT_DIR)
except FileNotFoundError:
    MODEL_PATH = (
        PROJECT_DIR
        / "runs"
        / "detect"
        / "runs"
        / "yolo26"
        / "full_training_from_corrected-2"
        / "weights"
        / "best.pt"
    )


st.set_page_config(
    page_title="SmearDx · Computer-Aided Hematology",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# CSS
# ------------------------------------------------------------------

CLINICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 Roboto, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(1200px 500px at 12% -10%, rgba(37, 99, 235, 0.16), transparent 55%),
        radial-gradient(900px 420px at 100% 0%, rgba(245, 158, 11, 0.07), transparent 50%),
        #0F172A;
    color: #E2E8F0;
}

.stApp header[data-testid="stHeader"] {
    background: rgba(15, 23, 42, 0.72);
    backdrop-filter: blur(12px);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header [data-testid="stToolbar"] { display: none; }

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 3.2rem;
    max-width: 1280px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #111827 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #E2E8F0 !important;
}

.hero {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
    padding: 18px 22px;
    margin-bottom: 18px;
    border-radius: 12px;
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 12px 40px rgba(2, 6, 23, 0.35);
    backdrop-filter: blur(16px);
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #93C5FD;
    background: rgba(37, 99, 235, 0.14);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 999px;
    padding: 5px 11px;
    margin-bottom: 10px;
}

.pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34D399;
    box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }
    70% { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
    100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

.hero h1 {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    margin: 0 0 6px 0;
    line-height: 1.15;
}

.hero p {
    margin: 0;
    color: #94A3B8;
    font-size: 14.5px;
    line-height: 1.5;
    max-width: 720px;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    font-weight: 600;
    color: #E2E8F0;
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    padding: 5px 10px;
}

.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.dot-rbc { background: #EF4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.55); }
.dot-wbc { background: #3B82F6; box-shadow: 0 0 8px rgba(59, 130, 246, 0.55); }
.dot-plt { background: #F59E0B; box-shadow: 0 0 8px rgba(245, 158, 11, 0.55); }

.sidebar-brand {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 10px;
}

.telemetry-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 14px 12px 14px;
    margin-bottom: 10px;
    box-shadow: 0 8px 24px rgba(2, 6, 23, 0.25);
}

.telemetry-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
}

.telemetry-value {
    font-size: 26px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.03em;
    margin: 4px 0 8px 0;
}

.telemetry-bar {
    height: 6px;
    border-radius: 99px;
    background: #1E293B;
    overflow: hidden;
    border: 1px solid #334155;
}

.telemetry-bar > span {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2563EB, #60A5FA);
}

.telemetry-delta {
    margin-top: 7px;
    font-size: 12px;
    color: #34D399;
    font-weight: 600;
}

.protocol-note {
    font-size: 12px;
    line-height: 1.55;
    color: #94A3B8;
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.22);
    border-radius: 12px;
    padding: 12px 13px;
    margin: 4px 0 14px 0;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 8px 0 18px 0;
}

@media (max-width: 980px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

.kpi-card {
    background: rgba(30, 41, 59, 0.72);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 16px 13px 16px;
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.28);
}

.kpi-card .kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #94A3B8;
    display: flex;
    align-items: center;
    gap: 7px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.kpi-card .kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.04em;
    margin-top: 6px;
}

.kpi-card .kpi-sub {
    font-size: 12px;
    color: #64748B;
    margin-top: 4px;
}

.kpi-rbc { box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.18), 0 10px 28px rgba(2, 6, 23, 0.28); }
.kpi-wbc { box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.18), 0 10px 28px rgba(2, 6, 23, 0.28); }
.kpi-plt { box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.18), 0 10px 28px rgba(2, 6, 23, 0.28); }

.panel {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 18px 8px 18px;
    margin-bottom: 16px;
}

.panel-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #CBD5E1;
    margin-bottom: 4px;
}

.panel-sub {
    font-size: 13px;
    color: #64748B;
    margin-bottom: 12px;
}

.upload-hint {
    border: 1px dashed rgba(148, 163, 184, 0.35);
    border-radius: 12px;
    padding: 18px 18px 6px 18px;
    background: rgba(15, 23, 42, 0.35);
    margin-bottom: 8px;
}

.image-frame {
    background: #020617;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 12px 32px rgba(2, 6, 23, 0.35);
}

.image-caption {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 0 0 8px 2px;
}

.empty-state {
    text-align: center;
    padding: 42px 20px;
    border-radius: 12px;
    border: 1px dashed #334155;
    background: rgba(15, 23, 42, 0.4);
    color: #94A3B8;
}

.audit-chip {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 999px;
}

.chip-ok {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.chip-no {
    background: rgba(239, 68, 68, 0.12);
    color: #FCA5A5;
    border: 1px solid rgba(239, 68, 68, 0.28);
}

div[data-testid="stTabs"] button {
    font-weight: 600;
    letter-spacing: 0.02em;
}

div[data-baseweb="tab-highlight"] {
    background-color: #3B82F6 !important;
}

.stButton > button,
.stDownloadButton > button {
    background: #2563EB;
    color: #F8FAFC;
    border: 1px solid rgba(147, 197, 253, 0.35);
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.28);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1D4ED8;
    border-color: #93C5FD;
}

[data-testid="stFileUploader"] {
    background: transparent;
}

[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(59, 130, 246, 0.35) !important;
    background: rgba(15, 23, 42, 0.45) !important;
    border-radius: 12px !important;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.35);
    border: 1px solid #334155;
    border-radius: 12px;
}

hr { border-color: #1E293B; }
</style>
"""


def inject_css() -> None:
    st.markdown(CLINICAL_CSS, unsafe_allow_html=True)


def render_html(markup: str) -> None:
    """Render HTML in Streamlit without it being escaped as text.

    Streamlit's markdown parser ends an HTML block at a blank line, so
    later sibling tags (like the 4th KPI card) would otherwise appear as
    literal ``<div class="kpi-card">`` on the page.
    """

    compact = "\n".join(line for line in markup.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def _font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _share(count: int, total: int) -> str:
    if total <= 0:
        return "0% of field"
    return f"{(count / total) * 100:.1f}% of field"


def _ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "—"
    return f"{numerator / denominator:.2f}"


def plotly_layout(title: str, height: int = 320, **overrides) -> dict:
    """Single layout dict for Plotly — nested keys (yaxis, xaxis) are merged.

    Never pass the same keyword both via this helper and as a second
    argument to ``Figure.update_layout``; that raises TypeError.
    """

    layout = {
        "title": dict(text=title, font=dict(size=14, color="#E2E8F0"), x=0),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.35)",
        "font": dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
        "margin": dict(l=40, r=16, t=48, b=40),
        "height": height,
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        "xaxis": dict(gridcolor="#1E293B", zeroline=False),
        "yaxis": dict(gridcolor="#1E293B", zeroline=False),
        "bargap": 0.28,
    }
    for key, value in overrides.items():
        existing = layout.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged = dict(existing)
            merged.update(value)
            layout[key] = merged
        else:
            layout[key] = value
    return layout


def apply_plotly_layout(fig, title: str, height: int = 320, **overrides):
    fig.update_layout(**plotly_layout(title, height=height, **overrides))
    return fig


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

if not MODEL_PATH.exists():
    inject_css()
    st.error(f"Final model could not be found at `{MODEL_PATH}`.")
    st.stop()


@st.cache_resource
def load_model():
    from ultralytics import YOLO
    from eval_config import assert_canonical_mapping

    loaded = YOLO(str(MODEL_PATH))
    assert_canonical_mapping(loaded.names)
    return loaded


inject_css()

try:
    model = load_model()
    ENGINE_STATUS = "System Ready"
except Exception as exc:
    model = None
    ENGINE_STATUS = "Engine fault"
    st.error(f"Could not load model: {exc}")
    st.stop()


# ------------------------------------------------------------------
# Layout helpers
# ------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <div class="hero-kicker">
              <span class="pulse"></span>
              {ENGINE_STATUS} · YOLO26n engine
            </div>
            <h1>SmearDx</h1>
            <p>
              Computer-aided hematology — localize, classify, and quantify
              RBCs, WBCs, and platelets from a single microscopic smear.
              Research prototype. Field-of-view counts are not a CBC.
            </p>
            <div class="pill-row">
              <span class="pill"><span class="dot dot-rbc"></span>RBC · Crimson</span>
              <span class="pill"><span class="dot dot-wbc"></span>WBC · Blue</span>
              <span class="pill"><span class="dot dot-plt"></span>Platelets · Amber</span>
              <span class="pill">Input {IMAGE_SIZE}×{IMAGE_SIZE}</span>
              <span class="pill">3-class YOLO26n</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def telemetry_card(label: str, value: str, width_pct: float, delta: str) -> str:
    width = max(0.0, min(100.0, width_pct))
    return f"""
    <div class="telemetry-card">
      <div class="telemetry-label">{label}</div>
      <div class="telemetry-value">{value}</div>
      <div class="telemetry-bar"><span style="width:{width:.1f}%"></span></div>
      <div class="telemetry-delta">{delta}</div>
    </div>
    """


def render_sidebar() -> Tuple[Dict[str, float], float]:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">Diagnostic telemetry</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="protocol-note">
              <b>Scientific evaluation ≠ live inference.</b><br>
              mAP@50 / precision below are COCO detector metrics at
              <code>conf={COCO_VAL_CONF}</code> on {TEST_IMAGE_COUNT} held-out
              images ({TEST_INSTANCE_COUNT} boxes). Counting uses the class
              gates you set here.
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_html(
            telemetry_card(
                "mAP@50  ·  COCO",
                f"{SCIENTIFIC_METRICS['map50']:.2f}%",
                SCIENTIFIC_METRICS["map50"],
                "+3.59 pts vs baseline",
            )
        )
        render_html(
            telemetry_card(
                "Precision  ·  COCO",
                f"{SCIENTIFIC_METRICS['precision']:.2f}%",
                SCIENTIFIC_METRICS["precision"],
                "+10.72 pts vs baseline",
            )
        )
        render_html(
            telemetry_card(
                "mAP@50–95  ·  COCO",
                f"{SCIENTIFIC_METRICS['map50_95']:.2f}%",
                SCIENTIFIC_METRICS["map50_95"],
                "+3.60 pts vs baseline",
            )
        )

        with st.expander("Live counting gates", expanded=True):
            st.caption(
                "YOLO is queried at the lowest gate, then each box is "
                "accepted or rejected per class. These sliders do not "
                "change reported mAP."
            )
            rbc_gate = st.slider(
                "RBC gate",
                min_value=0.10,
                max_value=0.95,
                value=float(CLASS_THRESHOLDS["RBC"]),
                step=0.05,
            )
            wbc_gate = st.slider(
                "WBC gate",
                min_value=0.10,
                max_value=0.95,
                value=float(CLASS_THRESHOLDS["WBC"]),
                step=0.05,
            )
            plt_gate = st.slider(
                "Platelet gate",
                min_value=0.10,
                max_value=0.95,
                value=float(CLASS_THRESHOLDS["Platelets"]),
                step=0.05,
            )

        with st.expander("Detector controls", expanded=False):
            iou = st.slider(
                "NMS IoU",
                min_value=0.30,
                max_value=0.90,
                value=DEFAULT_IOU,
                step=0.05,
                help="Higher IoU keeps more overlapping boxes (dense RBCs).",
            )
            st.caption(f"Inference image size locked at **{IMAGE_SIZE} px**.")
            st.caption(
                f"YOLO infer confidence = min gate "
                f"(currently {min(rbc_gate, wbc_gate, plt_gate):.0%})."
            )

        with st.expander("Model contract", expanded=False):
            st.caption("Canonical class mapping — do not invert.")
            for class_id, class_name in CANONICAL_CLASS_NAMES.items():
                st.code(f"{class_id} → {class_name}")
            st.caption("Weights")
            st.code(str(MODEL_PATH), language="text")

    gates = {"RBC": rbc_gate, "WBC": wbc_gate, "Platelets": plt_gate}
    return gates, iou


def kpi_card(class_key: str, title: str, value: str, subtitle: str) -> str:
    modifier = f" kpi-{class_key}" if class_key else ""
    dot = (
        f'<span class="dot dot-{class_key}"></span>'
        if class_key in {"rbc", "wbc", "plt"}
        else ""
    )
    return (
        f'<div class="kpi-card{modifier}">'
        f'<div class="kpi-label">{dot}{title}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{subtitle}</div>'
        f"</div>"
    )


def render_kpi_row(counts: Dict[str, int], rejected: int, avg_conf: float) -> None:
    total = sum(counts.values())
    rbc, wbc, plt = counts["RBC"], counts["WBC"], counts["Platelets"]
    summary_sub = (
        f"RBC:WBC {_ratio(rbc, wbc)} · PLT/100 RBC {_ratio(plt * 100, rbc)} "
        f"· rejected {rejected} · mean conf {avg_conf:.0%}"
    )
    cards = [
        kpi_card("rbc", "RBC", str(rbc), _share(rbc, total)),
        kpi_card("wbc", "WBC", str(wbc), _share(wbc, total)),
        kpi_card("plt", "Platelets", str(plt), _share(plt, total)),
        kpi_card("", "Field summary", str(total), summary_sub),
    ]
    columns = st.columns(4, gap="small")
    for column, markup in zip(columns, cards):
        with column:
            render_html(markup)


def annotate_image(image: Image.Image, detections: list) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    label_font = _font(14)

    for detection in detections:
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["box"]
        color = UI_BOX_COLORS.get(class_name, (148, 163, 184))

        draw.rectangle(
            [int(x1), int(y1), int(x2), int(y2)],
            outline=color,
            width=3,
        )

        label = f"{class_name} {confidence:.2f}"
        text_x = int(x1)
        text_y = max(0, int(y1) - 18)
        try:
            bbox = draw.textbbox((text_x, text_y), label, font=label_font)
            draw.rectangle(bbox, fill=color)
        except Exception:
            pass
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=label_font)

    return annotated


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_inference(
    image: Image.Image,
    gates: Dict[str, float],
    iou: float,
) -> Tuple[Image.Image, Dict[str, int], list, list, float]:
    min_conf = min(gates.values())
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        results = model.predict(
            source=temp_path,
            imgsz=IMAGE_SIZE,
            conf=min_conf,
            iou=iou,
            verbose=False,
        )
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    result = results[0]
    accepted: list = []
    rejected: list = []

    if result.boxes is not None and len(result.boxes) > 0:
        for index in range(len(result.boxes)):
            class_id = int(result.boxes.cls[index])
            confidence = float(result.boxes.conf[index])
            class_name = model.names[class_id]
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": result.boxes.xyxy[index].tolist(),
            }
            required = gates.get(class_name, min_conf)
            if confidence >= required:
                accepted.append(detection)
            else:
                rejected.append(detection)

    counts = empty_class_counts()
    confidences: List[float] = []
    for detection in accepted:
        class_name = detection["class_name"]
        if class_name not in counts:
            continue
        counts[class_name] += 1
        confidences.append(detection["confidence"])

    average_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )
    annotated = annotate_image(image, accepted)
    return annotated, counts, accepted, rejected, average_confidence


def render_live_tab(gates: Dict[str, float], iou: float) -> None:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">Specimen intake</div>
          <div class="panel-sub">
            Drop a Wright–Giemsa or similar smear. Supported formats:
            JPG, JPEG, PNG, BMP. Live counts use the sidebar gates, not COCO mAP.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload microscopic blood smear",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="empty-state">
              <div style="font-size:28px;margin-bottom:8px;">◎</div>
              <div style="color:#E2E8F0;font-weight:700;margin-bottom:6px;">
                Awaiting smear image
              </div>
              Detector mAP and counting MAE remain on the Evaluation tab
              until a field is analysed.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running YOLO26n at the live operating point…"):
        try:
            annotated, counts, accepted, rejected, avg_conf = run_inference(
                image, gates, iou
            )
        except Exception as exc:
            st.error(f"Error while processing image: {exc}")
            return

    render_kpi_row(counts, len(rejected), avg_conf)

    original_col, detected_col = st.columns([1, 1], gap="large")
    with original_col:
        st.markdown(
            '<div class="image-caption">Original field</div>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
    with detected_col:
        st.markdown(
            '<div class="image-caption">Annotated overlay · live gates</div>',
            unsafe_allow_html=True,
        )
        st.image(annotated, use_container_width=True)

    download_col, meta_col = st.columns([1, 2])
    with download_col:
        st.download_button(
            "Download annotated PNG",
            data=image_to_png_bytes(annotated),
            file_name="smeardx_annotated.png",
            mime="image/png",
            use_container_width=True,
        )
    with meta_col:
        st.caption(
            f"Accepted {len(accepted)} · rejected {len(rejected)} · "
            f"gates RBC {gates['RBC']:.0%} / WBC {gates['WBC']:.0%} / "
            f"PLT {gates['Platelets']:.0%} · NMS IoU {iou:.2f} · "
            "not a CBC or WBC differential."
        )

    with st.expander("Detection audit log", expanded=False):
        if not accepted and not rejected:
            st.write("No detections returned at the minimum YOLO confidence.")
        else:
            rows = []
            for detection in accepted + rejected:
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                required = gates.get(class_name, min(gates.values()))
                status = "Accepted" if confidence >= required else "Rejected"
                rows.append(
                    {
                        "Cell type": class_name,
                        "Confidence": f"{confidence:.1%}",
                        "Gate": f"{required:.0%}",
                        "Status": status,
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
            )


def render_accuracy_tab() -> None:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">Quantitative accuracy</div>
          <div class="panel-sub">
            Protocol A is COCO mAP at conf=0.001. Protocol B is count MAE/MAPE
            at the frozen paper gates (RBC 0.60 / WBC 0.40 / PLT 0.40), not
            the live sliders.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall mAP@50", f"{SCIENTIFIC_METRICS['map50']:.2f}%", "+3.59 pts")
    with c2:
        st.metric("Precision", f"{SCIENTIFIC_METRICS['precision']:.2f}%", "+10.72 pts")
    with c3:
        st.metric("mAP@50–95", f"{SCIENTIFIC_METRICS['map50_95']:.2f}%", "+3.60 pts")

    classes = list(DISPLAY_CLASS_ORDER)
    map50 = [SCIENTIFIC_METRICS["per_class_map50"][name] for name in classes]
    precision = [SCIENTIFIC_METRICS["per_class_precision"][name] for name in classes]
    colors = [UI_HEX[name] for name in classes]

    if PLOTLY_AVAILABLE:
        try:
            fig = go.Figure()
            fig.add_bar(
                name="mAP@50",
                x=classes,
                y=map50,
                marker_color=colors,
                opacity=0.95,
            )
            fig.add_bar(
                name="Precision",
                x=classes,
                y=precision,
                marker_color=colors,
                opacity=0.45,
            )
            apply_plotly_layout(
                fig,
                "Per-class COCO detector metrics (%)",
                barmode="group",
                yaxis=dict(range=[0, 100], ticksuffix="%"),
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_bar(
                name="Baseline",
                x=["mAP@50", "Precision", "mAP@50–95"],
                y=[
                    BASELINE_METRICS["map50"],
                    BASELINE_METRICS["precision"],
                    BASELINE_METRICS["map50_95"],
                ],
                marker_color="#64748B",
            )
            fig2.add_bar(
                name="Fine-tuned",
                x=["mAP@50", "Precision", "mAP@50–95"],
                y=[
                    SCIENTIFIC_METRICS["map50"],
                    SCIENTIFIC_METRICS["precision"],
                    SCIENTIFIC_METRICS["map50_95"],
                ],
                marker_color="#3B82F6",
            )
            apply_plotly_layout(
                fig2,
                "Baseline vs fine-tuned (COCO test, 36 images)",
                barmode="group",
                yaxis=dict(range=[0, 100], ticksuffix="%"),
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as exc:
            st.warning(f"Evaluation charts could not be rendered: {exc}")
    else:
        st.info("Install `plotly` to render evaluation charts. Tables remain below.")

    detection_df = pd.DataFrame(
        {
            "Cell type": classes,
            "mAP@50 (COCO)": [f"{value:.1f}%" for value in map50],
            "Precision (COCO)": [f"{value:.1f}%" for value in precision],
            "Paper counting gate": [
                f"{CLASS_THRESHOLDS[name]:.0%}" for name in classes
            ],
        }
    )
    st.dataframe(detection_df, hide_index=True, use_container_width=True)

    st.markdown("#### Counting error · operating point")
    count_metrics = load_count_metrics()
    if count_metrics is None:
        st.warning(
            "Counting MAE/MAPE is not on disk yet. Run "
            "`python evaluate_quick_test.py --skip-coco --no-save-predictions` "
            "to write `runs/yolo26_evaluation/count_metrics.json`."
        )
        return

    class_stats = count_metrics.get("classes", {})
    overall = count_metrics.get("overall", {})
    mae_cols = st.columns(4)
    labels = list(DISPLAY_CLASS_ORDER) + ["Overall"]
    values = [class_stats.get(name, {}) for name in DISPLAY_CLASS_ORDER] + [overall]
    for column, label, stats in zip(mae_cols, labels, values):
        with column:
            st.metric(f"{label} MAE", format_metric(stats.get("mae")))
            st.caption(
                f"MAPE {format_metric(stats.get('mape'), '%')} · "
                f"n={stats.get('n_mape', 0)} · "
                f"zero-GT skipped={stats.get('n_zero_gt', 0)}"
            )

    if PLOTLY_AVAILABLE:
        try:
            mae_vals = [
                class_stats.get(name, {}).get("mae") or 0 for name in classes
            ]
            gt_totals = [
                class_stats.get(name, {}).get("gt_total") or 0 for name in classes
            ]
            pred_totals = [
                class_stats.get(name, {}).get("pred_total") or 0 for name in classes
            ]
            fig3 = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Count MAE (cells / image)", "GT vs predicted totals"),
            )
            fig3.add_trace(
                go.Bar(
                    x=classes,
                    y=mae_vals,
                    marker_color=colors,
                    name="MAE",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig3.add_trace(
                go.Bar(
                    x=classes,
                    y=gt_totals,
                    name="Ground truth",
                    marker_color="#64748B",
                ),
                row=1,
                col=2,
            )
            fig3.add_trace(
                go.Bar(
                    x=classes,
                    y=pred_totals,
                    name="Predicted",
                    marker_color="#3B82F6",
                ),
                row=1,
                col=2,
            )
            apply_plotly_layout(
                fig3,
                "Operating-point count accuracy",
                height=340,
                barmode="group",
            )
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as exc:
            st.warning(f"Count-error charts could not be rendered: {exc}")

    count_rows = []
    for class_name in DISPLAY_CLASS_ORDER:
        stats = class_stats.get(class_name, {})
        count_rows.append(
            {
                "Cell type": class_name,
                "GT total": stats.get("gt_total", "—"),
                "Pred total": stats.get("pred_total", "—"),
                "MAE": format_metric(stats.get("mae")),
                "MAPE": format_metric(stats.get("mape"), "%"),
                "Images in MAPE": stats.get("n_mape", 0),
                "Zero-GT skipped": stats.get("n_zero_gt", 0),
            }
        )
    st.dataframe(pd.DataFrame(count_rows), hide_index=True, use_container_width=True)

    per_image = count_metrics.get("per_image") or []
    if per_image:
        with st.expander("Per-image count table"):
            rows = []
            for row in per_image:
                gt = row.get("gt", {})
                pred = row.get("pred", {})
                rows.append(
                    {
                        "Image": row.get("image"),
                        "GT WBC": gt.get("WBC"),
                        "Pred WBC": pred.get("WBC"),
                        "GT RBC": gt.get("RBC"),
                        "Pred RBC": pred.get("RBC"),
                        "GT PLT": gt.get("Platelets"),
                        "Pred PLT": pred.get("Platelets"),
                        "Empty GT": bool(row.get("zero_gt")),
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main() -> None:
    render_header()
    gates, iou = render_sidebar()
    live_tab, accuracy_tab = st.tabs(["Live inference", "Quantitative accuracy"])
    with live_tab:
        render_live_tab(gates, iou)
    with accuracy_tab:
        render_accuracy_tab()


main()
