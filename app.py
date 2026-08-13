import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import tempfile
import os
import pandas as pd


# ============================================================
# BLOOD CELL AI DETECTOR
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = (
        PROJECT_DIR
        / "runs"
        / "detect"
        / "runs"
        / "yolo26"
        / "corrected_20_retrain-3"
        / "weights"
        / "best.pt"
)

CONFIDENCE = 0.50
IMAGE_SIZE = 320


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Blood Cell AI Detector",
    page_icon="🩸",
    layout="wide"
)


# ============================================================
# SIMPLE UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .subtitle {
        font-size: 18px;
        color: #6c757d;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    .section-note {
        color: #6c757d;
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 18px;
    }

    .result-summary {
        padding: 18px 22px;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.08);
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.7;
    }

    .improvement-box {
        padding: 18px 22px;
        border-radius: 12px;
        background-color: rgba(40, 167, 69, 0.08);
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        f"""
        ⚠️ YOLO26 model was not found.

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
    st.error(f"Could not load YOLO26 model: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🩸 Blood Cell AI Detector")

st.markdown(
    """
    <div class="subtitle">
        AI-powered microscopic blood smear analysis using YOLO26
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Upload a microscopic blood smear image to automatically detect "
    "and count **Red Blood Cells (RBCs), White Blood Cells (WBCs), "
    "and Platelets**."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Model")

    st.write("**YOLO26n Blood Cell Detector**")

    st.divider()

    st.caption("Prediction Settings")

    st.write(f"Confidence threshold: **{CONFIDENCE:.0%}**")
    st.write(f"Input image size: **{IMAGE_SIZE}px**")

    st.divider()

    st.caption("Detected Classes")

    st.write("🔵 **WBC** — White Blood Cells")
    st.write("🔴 **RBC** — Red Blood Cells")
    st.write("🟡 **Platelets**")

    st.divider()

    with st.expander("Technical Details"):

        st.write("Model path:")
        st.code(str(MODEL_PATH))

        st.write("Class mapping:")

        for class_id, class_name in model.names.items():
            st.write(f"{class_id} → {class_name}")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.divider()

st.subheader("📤 Upload Blood Smear Image")

uploaded_file = st.file_uploader(
    "Select a JPG, JPEG, PNG, or BMP microscopic image",
    type=["jpg", "jpeg", "png", "bmp"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    temp_path = None

    try:

        # ----------------------------------------------------
        # SAVE TEMPORARY IMAGE
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

        with st.spinner("🔬 Analysing blood smear..."):

            results = model.predict(
                source=temp_path,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                verbose=False
            )

        result = results[0]


        # ====================================================
        # COUNT DETECTED CELLS
        # ====================================================

        counts = {
            "WBC": 0,
            "RBC": 0,
            "Platelets": 0
        }

        confidence_values = []

        if result.boxes is not None and len(result.boxes) > 0:

            for i in range(len(result.boxes)):

                class_id = int(result.boxes.cls[i])
                confidence = float(result.boxes.conf[i])

                class_name = model.names[class_id]

                if class_name in counts:

                    counts[class_name] += 1
                    confidence_values.append(confidence)


        total_cells = sum(counts.values())


        # ====================================================
        # AVERAGE DETECTION CONFIDENCE
        # ====================================================

        if confidence_values:

            average_confidence = (
                    sum(confidence_values)
                    / len(confidence_values)
            )

        else:

            average_confidence = 0


        # ====================================================
        # CREATE ANNOTATED IMAGE
        # ====================================================

        annotated_image = result.plot()

        # YOLO returns BGR.
        # Streamlit expects RGB.
        annotated_image = annotated_image[:, :, ::-1]


        # ====================================================
        # SIDE-BY-SIDE IMAGES
        # ====================================================

        st.divider()

        st.subheader("🔬 Detection Results")

        st.markdown(
            """
            <div class="section-note">
                Compare the original microscope image with the
                AI-detected image for visual verification.
            </div>
            """,
            unsafe_allow_html=True
        )


        original_col, detected_col = st.columns(
            2,
            gap="large"
        )


        with original_col:

            st.markdown("#### Original Image")

            st.image(
                image,
                caption="Uploaded blood smear",
                use_container_width=True
            )


        with detected_col:

            st.markdown("#### AI Detected Image")

            st.image(
                annotated_image,
                caption="YOLO26 detected cells",
                use_container_width=True
            )


        # ====================================================
        # CELL COUNTS
        # ====================================================

        st.markdown("### 📊 Detected Cell Counts")


        wbc_col, rbc_col, platelet_col, total_col = st.columns(4)


        with wbc_col:

            st.metric(
                label="🔵 WBC",
                value=counts["WBC"]
            )


        with rbc_col:

            st.metric(
                label="🔴 RBC",
                value=counts["RBC"]
            )


        with platelet_col:

            st.metric(
                label="🟡 Platelets",
                value=counts["Platelets"]
            )


        with total_col:

            st.metric(
                label="🩸 Total Cells",
                value=total_cells
            )


        if confidence_values:

            st.caption(
                f"Average confidence across detected cells: "
                f"**{average_confidence:.1%}**  •  "
                f"Detection threshold: **{CONFIDENCE:.0%}**"
            )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.divider()

        st.subheader("🎯 Model Accuracy & Detection Performance")

        st.markdown(
            """
            <div class="section-note">
                Evaluation results measured on the model validation/test
                dataset.
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # HEADLINE METRICS
        # ----------------------------------------------------

        metric1, metric2, metric3 = st.columns(3)


        with metric1:

            st.metric(
                label="Overall mAP@50",
                value="81.74%"
            )


        with metric2:

            st.metric(
                label="Precision",
                value="78.50%",
                delta="+6.76 pts"
            )


        with metric3:

            st.metric(
                label="WBC mAP@50",
                value="96.60%"
            )


        # ====================================================
        # MODEL PERFORMANCE TABLE
        # ====================================================

        performance_data = {
            "Metric": [
                "Overall mAP@50",
                "Precision",
                "WBC mAP@50",
                "RBC mAP@50",
                "Platelet mAP@50"
            ],

            "Result": [
                "81.74%",
                "78.50%",
                "96.60%",
                "78.20%",
                "70.50%"
            ],

            "Assessment": [
                "🟢 Strong",
                "🟢 High Reliability",
                "🟢 Excellent",
                "🟢 Strong",
                "🟢 Good"
            ]
        }


        performance_df = pd.DataFrame(
            performance_data
        )


        st.dataframe(
            performance_df,
            hide_index=True,
            use_container_width=True
        )


        # ====================================================
        # RETRAINING IMPROVEMENTS
        # ====================================================

        st.markdown("### 🚀 Impact of Targeted Retraining")


        improvement_col1, improvement_col2 = st.columns(2)


        with improvement_col1:

            st.metric(
                label="Precision",
                value="78.50%",
                delta="+6.76 percentage points"
            )

            st.caption(
                "Baseline: 71.74% → Retrained: 78.50%"
            )


        with improvement_col2:

            st.metric(
                label="Platelet mAP@50",
                value="70.50%",
                delta="+5.40 percentage points"
            )

            st.caption(
                "Baseline: 65.10% → Retrained: 70.50%"
            )


        # ====================================================
        # KEY RESULT
        # ====================================================

        st.markdown(
            """
            <div class="result-summary">

            <b>Key Result</b><br><br>

            The retrained model achieved <b>81.74% overall mAP@50</b>
            with <b>78.50% precision</b>.

            WBC detection remained exceptionally strong at
            <b>96.60% mAP@50</b>, while targeted retraining improved
            platelet detection from <b>65.10% → 70.50%</b> and
            overall precision from <b>71.74% → 78.50%</b>.

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # VISUAL VERIFICATION
        # ====================================================

        st.divider()

        st.subheader("👁️ Visual Verification")

        st.write(
            "The side-by-side comparison allows direct visual verification "
            "of cell localization and classification. Bounding boxes can be "
            "checked against the original microscopic image to confirm that "
            "detected RBCs, WBCs, and Platelets correspond to visible cells."
        )


        st.success(
            f"""
            Analysis complete — **{total_cells} cells detected**

            🔵 **{counts["WBC"]} WBC**  •
            🔴 **{counts["RBC"]} RBC**  •
            🟡 **{counts["Platelets"]} Platelets**
            """
        )


    except Exception as e:

        st.error(
            f"❌ Error while processing image: {e}"
        )


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY FILE
        # ----------------------------------------------------

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
        "👆 Upload a microscopic blood smear image to start the analysis."
    )