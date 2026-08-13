import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import pandas as pd


# ============================================================
# BLOOD CELL AI DETECTOR
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================
# FINAL MODEL
# ============================================================

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


# ============================================================
# CLASS-SPECIFIC CONFIDENCE THRESHOLDS
# ============================================================
#
# YOLO first returns predictions using the LOWEST threshold.
# We then apply a separate acceptance threshold for each class.
#
# These are currently experimental application thresholds.
# They are NOT the thresholds used to calculate mAP.
# ============================================================

CLASS_THRESHOLDS = {
    "RBC": 0.60,
    "WBC": 0.40,
    "Platelets": 0.40,
}

MIN_CONFIDENCE = min(CLASS_THRESHOLDS.values())

IMAGE_SIZE = 640


# ============================================================
# FINAL TEST-SET PERFORMANCE
# 36 unseen test images
# 471 annotated instances
# ============================================================

FINAL_MAP50 = 85.40
FINAL_MAP5095 = 60.07
FINAL_PRECISION = 82.46

RBC_MAP50 = 85.40
WBC_MAP50 = 96.90
PLATELET_MAP50 = 73.90

RBC_PRECISION = 76.10
WBC_PRECISION = 97.20
PLATELET_PRECISION = 74.20


# ============================================================
# ORIGINAL BASELINE
# ============================================================

BASELINE_MAP50 = 81.81
BASELINE_MAP5095 = 56.47
BASELINE_PRECISION = 71.74

BASELINE_RBC_MAP50 = 83.40
BASELINE_WBC_MAP50 = 96.90
BASELINE_PLATELET_MAP50 = 65.10

BASELINE_RBC_PRECISION = 61.00
BASELINE_WBC_PRECISION = 95.00
BASELINE_PLATELET_PRECISION = 59.20


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Blood Cell AI Detector",
    page_icon="🩸",
    layout="wide",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    .subtitle {
        font-size: 18px;
        color: #6c757d;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    .info-box {
        padding: 14px 18px;
        border-radius: 10px;
        background-color: rgba(0, 123, 255, 0.06);
        margin-bottom: 20px;
    }

    .result-box {
        padding: 18px 22px;
        border-radius: 12px;
        background-color: rgba(40, 167, 69, 0.08);
        border-left: 5px solid #28a745;
        margin-top: 18px;
        line-height: 1.7;
    }

    .threshold-box {
        padding: 12px 16px;
        border-radius: 10px;
        background-color: rgba(255, 193, 7, 0.08);
        margin-top: 8px;
        margin-bottom: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        f"""
        Final model could not be found.

        Expected location:

        `{MODEL_PATH}`
        """
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()

except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🩸 Blood Cell AI Detector")

st.markdown(
    """
    <div class="subtitle">
        AI-powered microscopic blood smear analysis using YOLO26n
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="info-box">

    <b>Final Retrained YOLO26n Model</b><br>

    Detects and counts
    <b>Red Blood Cells (RBCs)</b>,
    <b>White Blood Cells (WBCs)</b>
    and <b>Platelets</b>.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Final Model")

    st.success("Full-Dataset Retrained Model")

    st.metric(
        "mAP@50",
        f"{FINAL_MAP50:.2f}%"
    )

    st.metric(
        "Precision",
        f"{FINAL_PRECISION:.2f}%"
    )

    st.divider()

    st.subheader("🎯 Class Thresholds")

    st.write(
        f"🔴 RBC: **{CLASS_THRESHOLDS['RBC']:.0%}**"
    )

    st.write(
        f"🔵 WBC: **{CLASS_THRESHOLDS['WBC']:.0%}**"
    )

    st.write(
        f"🟡 Platelets: **{CLASS_THRESHOLDS['Platelets']:.0%}**"
    )

    st.caption(
        "Class-specific thresholds control which "
        "predictions are displayed by the application."
    )

    st.divider()

    st.caption("Inference Settings")

    st.write(
        f"Image size: **{IMAGE_SIZE}px**"
    )

    st.write(
        f"Minimum YOLO confidence: "
        f"**{MIN_CONFIDENCE:.0%}**"
    )

    st.divider()

    with st.expander("Technical Details"):

        st.write("Model path:")

        st.code(str(MODEL_PATH))

        st.write("YOLO class mapping:")

        for class_id, class_name in model.names.items():

            st.write(
                f"{class_id} → {class_name}"
            )


# ============================================================
# FILE UPLOAD
# ============================================================

st.divider()

st.subheader("📤 Upload Blood Smear Image")

uploaded_file = st.file_uploader(
    "Select a microscopic blood smear image",
    type=["jpg", "jpeg", "png", "bmp"],
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    temp_path = None

    try:

        # ----------------------------------------------------
        # SAVE TEMP IMAGE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False
        ) as temp_file:

            image.save(temp_file.name)

            temp_path = temp_file.name


        # ====================================================
        # RUN YOLO
        # ====================================================

        with st.spinner(
                "🔬 Analysing blood smear..."
        ):

            results = model.predict(
                source=temp_path,
                imgsz=IMAGE_SIZE,

                # Important:
                # YOLO returns all candidates >= lowest
                # class-specific threshold.
                conf=MIN_CONFIDENCE,

                verbose=False,
            )


        result = results[0]


        # ====================================================
        # FILTER DETECTIONS BY CLASS
        # ============================================================

        accepted_detections = []

        rejected_detections = []


        if (
                result.boxes is not None
                and len(result.boxes) > 0
        ):

            for i in range(len(result.boxes)):

                class_id = int(
                    result.boxes.cls[i]
                )

                confidence = float(
                    result.boxes.conf[i]
                )

                class_name = model.names[
                    class_id
                ]

                threshold = CLASS_THRESHOLDS.get(
                    class_name,
                    MIN_CONFIDENCE
                )


                box = result.boxes.xyxy[i].tolist()


                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": box,
                }


                if confidence >= threshold:

                    accepted_detections.append(
                        detection
                    )

                else:

                    rejected_detections.append(
                        detection
                    )


        # ====================================================
        # COUNT ACCEPTED DETECTIONS
        # ============================================================

        counts = {
            "WBC": 0,
            "RBC": 0,
            "Platelets": 0,
        }


        class_confidences = {
            "WBC": [],
            "RBC": [],
            "Platelets": [],
        }


        all_confidences = []


        for detection in accepted_detections:

            class_name = detection[
                "class_name"
            ]

            confidence = detection[
                "confidence"
            ]


            if class_name in counts:

                counts[class_name] += 1

                class_confidences[
                    class_name
                ].append(
                    confidence
                )

                all_confidences.append(
                    confidence
                )


        total_cells = sum(
            counts.values()
        )


        # ====================================================
        # AVERAGE CONFIDENCE
        # ============================================================

        if all_confidences:

            average_confidence = (
                    sum(all_confidences)
                    / len(all_confidences)
            )

        else:

            average_confidence = 0


        # ====================================================
        # CREATE CUSTOM ANNOTATED IMAGE
        #
        # We draw ONLY detections that passed their
        # class-specific threshold.
        # ============================================================

        annotated_image = image.copy()

        draw = ImageDraw.Draw(
            annotated_image
        )


        # Colors:
        # WBC       = Blue
        # RBC       = Red
        # Platelets = Yellow

        CLASS_COLORS = {
            "WBC": (0, 102, 255),
            "RBC": (255, 60, 60),
            "Platelets": (255, 200, 0),
        }


        for detection in accepted_detections:

            class_name = detection[
                "class_name"
            ]

            confidence = detection[
                "confidence"
            ]

            x1, y1, x2, y2 = detection[
                "box"
            ]


            color = CLASS_COLORS.get(
                class_name,
                (0, 255, 0)
            )


            # Bounding box

            draw.rectangle(
                [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],
                outline=color,
                width=3,
            )


            # Label

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            text_x = int(x1)

            text_y = max(
                0,
                int(y1) - 18
            )


            # Label background

            try:

                bbox = draw.textbbox(
                    (text_x, text_y),
                    label
                )

                draw.rectangle(
                    bbox,
                    fill=color
                )

            except Exception:

                pass


            draw.text(
                (text_x, text_y),
                label,
                fill=(255, 255, 255),
            )


        # ====================================================
        # DETECTION RESULTS
        # ============================================================

        st.divider()

        st.subheader(
            "🔬 Detection Results"
        )


        st.markdown(
            """
            Compare the original microscope image with
            the AI-detected image for visual verification.
            """
        )


        # ====================================================
        # SIDE-BY-SIDE IMAGES
        # ============================================================

        original_col, detected_col = st.columns(
            2,
            gap="large"
        )


        with original_col:

            st.markdown(
                "#### Original Image"
            )

            st.image(
                image,
                caption="Uploaded blood smear",
                width=500,
            )


        with detected_col:

            st.markdown(
                "#### AI Detected Image"
            )

            st.image(
                annotated_image,
                caption=(
                    "Final YOLO26n prediction "
                    "with class-specific thresholds"
                ),
                width=500,
            )


        # ====================================================
        # CELL COUNTS
        # ============================================================

        st.markdown(
            "### 📊 Predicted Cell Counts"
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "🔵 WBC",
                counts["WBC"]
            )


        with col2:

            st.metric(
                "🔴 RBC",
                counts["RBC"]
            )


        with col3:

            st.metric(
                "🟡 Platelets",
                counts["Platelets"]
            )


        with col4:

            st.metric(
                "🩸 Total Cells",
                total_cells
            )


        # ====================================================
        # CONFIDENCE SUMMARY
        # ============================================================

        if all_confidences:

            st.caption(
                f"Average confidence of accepted detections: "
                f"**{average_confidence:.1%}**"
            )

        # ====================================================
        # FINAL MODEL PERFORMANCE
        # ============================================================

        st.divider()

        st.subheader(
            "🎯 Final Model Performance"
        )


        st.caption(
            "Evaluated on 36 unseen test images "
            "containing 471 annotated cell instances."
        )


        metric1, metric2, metric3 = (
            st.columns(3)
        )


        with metric1:

            st.metric(
                "Overall mAP@50",
                f"{FINAL_MAP50:.2f}%",
                "+3.59 pts"
            )


        with metric2:

            st.metric(
                "Precision",
                f"{FINAL_PRECISION:.2f}%",
                "+10.72 pts"
            )


        with metric3:

            st.metric(
                "mAP@50–95",
                f"{FINAL_MAP5095:.2f}%",
                "+3.60 pts"
            )


        st.caption(
            "Improvement shown relative to the "
            "original full-dataset model."
        )


        # ====================================================
        # CLASS PERFORMANCE
        # ============================================================

        st.markdown(
            "### 🧬 Performance by Cell Type"
        )


        performance_df = pd.DataFrame(
            {
                "Cell Type": [
                    "🔵 WBC",
                    "🔴 RBC",
                    "🟡 Platelets",
                ],

                "mAP@50": [
                    f"{WBC_MAP50:.1f}%",
                    f"{RBC_MAP50:.1f}%",
                    f"{PLATELET_MAP50:.1f}%",
                ],

                "Precision": [
                    f"{WBC_PRECISION:.1f}%",
                    f"{RBC_PRECISION:.1f}%",
                    f"{PLATELET_PRECISION:.1f}%",
                ],

                "App Threshold": [
                    f"{CLASS_THRESHOLDS['WBC']:.0%}",
                    f"{CLASS_THRESHOLDS['RBC']:.0%}",
                    f"{CLASS_THRESHOLDS['Platelets']:.0%}",
                ],

                "Assessment": [
                    "🟢 Excellent",
                    "🟢 Strong",
                    "🟢 Strong",
                ],
            }
        )


        st.dataframe(
            performance_df,
            hide_index=True,
            use_container_width=True,
        )


        # ====================================================
        # BASELINE VS FINAL
        # ============================================================

        st.markdown(
            "### 🚀 Model Improvement"
        )


        improvement_df = pd.DataFrame(
            {
                "Metric": [
                    "Overall mAP@50",
                    "mAP@50–95",
                    "Overall Precision",
                    "RBC mAP@50",
                    "RBC Precision",
                    "Platelet mAP@50",
                    "Platelet Precision",
                    "WBC mAP@50",
                    "WBC Precision",
                ],

                "Original": [
                    "81.81%",
                    "56.47%",
                    "71.74%",
                    "83.40%",
                    "61.00%",
                    "65.10%",
                    "59.20%",
                    "96.90%",
                    "95.00%",
                ],

                "Final Retrained": [
                    "85.40%",
                    "60.07%",
                    "82.46%",
                    "85.40%",
                    "76.10%",
                    "73.90%",
                    "74.20%",
                    "96.90%",
                    "97.20%",
                ],

                "Improvement": [
                    "+3.59 pts",
                    "+3.60 pts",
                    "+10.72 pts",
                    "+2.00 pts",
                    "+15.10 pts",
                    "+8.80 pts",
                    "+15.00 pts",
                    "Maintained",
                    "+2.20 pts",
                ],
            }
        )


        st.dataframe(
            improvement_df,
            hide_index=True,
            use_container_width=True,
        )


        # ====================================================
        # FINAL SUMMARY
        # ============================================================

        st.markdown(
            f"""
            <div class="result-box">

            <b>🏆 Final Result</b><br><br>

            The final retrained YOLO26n achieved
            <b>85.40% mAP@50</b> and
            <b>82.46% precision</b> on the
            independent 36-image test set.

            <br><br>

            For this image, the model detected:

            <br>

            🔵 <b>{counts["WBC"]} WBC</b><br>
            🔴 <b>{counts["RBC"]} RBC</b><br>
            🟡 <b>{counts["Platelets"]} Platelets</b><br>

            <br>

            Total:
            <b>{total_cells} detected cells</b>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ====================================================
        # OPTIONAL DEBUG INFORMATION
        # ============================================================

        with st.expander(
                "🔍 View detection confidence details"
        ):

            if accepted_detections:

                detection_rows = []

                for detection in accepted_detections:

                    detection_rows.append(
                        {
                            "Cell Type":
                                detection["class_name"],

                            "Confidence":
                                f"{detection['confidence']:.1%}",

                            "Required Threshold":
                                f"{CLASS_THRESHOLDS.get(detection['class_name'], MIN_CONFIDENCE):.0%}",

                            "Status":
                                "Accepted",
                        }
                    )


                detection_df = pd.DataFrame(
                    detection_rows
                )


                st.dataframe(
                    detection_df,
                    hide_index=True,
                    use_container_width=True,
                )

            else:

                st.write(
                    "No detections passed the "
                    "class-specific thresholds."
                )


    except Exception as e:

        st.error(
            f"Error while processing image: {e}"
        )


    finally:

        if temp_path is not None:

            try:
                os.remove(temp_path)

            except Exception:
                pass


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "👆 Upload a microscopic blood smear image "
        "to start the analysis."
    )