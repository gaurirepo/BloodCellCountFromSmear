import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import tempfile
import os

# ============================================================
# BLOOD CELL AI DETECTOR
# YOLO26
# ============================================================

# ------------------------------------------------------------
# PROJECT DIRECTORY
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# CURRENTLY USE QUICK-TEST MODEL
# ------------------------------------------------------------
# Use this while testing the app BEFORE full training.
# ------------------------------------------------------------

MODEL_PATH = (
        PROJECT_DIR
        / "runs"
        / "detect"
        / "runs"
        / "yolo26"
        / "quick_test"
        / "weights"
        / "best.pt"
)

# ------------------------------------------------------------
# AFTER FULL TRAINING, CHANGE TO:
#
# MODEL_PATH = (
#     PROJECT_DIR
#     / "runs"
#     / "detect"
#     / "runs"
#     / "yolo26"
#     / "full_training"
#     / "weights"
#     / "best.pt"
# )
# ------------------------------------------------------------


CONFIDENCE = 0.50
IMAGE_SIZE = 640


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Blood Cell AI Detector",
    page_icon="🩸",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩸 Blood Cell AI Detector")

st.markdown(
    """
    ### YOLO26 Microscopic Blood Smear Analysis

    Upload **any original microscope image** of a blood smear.

    The AI will automatically:

    - 🔴 Detect and count RBCs
    - ⚪ Detect and count WBCs
    - 🟣 Detect and count Platelets
    - 🔲 Draw bounding boxes around detected cells
    - 📊 Display the total number of detected cells
    """
)

st.divider()


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        f"""
        ⚠️ YOLO26 model was not found.

        Expected location:

        `{MODEL_PATH}`

        Please make sure the model exists.
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
# SIDEBAR
# ============================================================

st.sidebar.header("🤖 Model Information")

st.sidebar.write("Model: YOLO26n")

st.sidebar.write(
    f"Confidence threshold: {CONFIDENCE}"
)

st.sidebar.write(
    f"Image size: {IMAGE_SIZE}"
)

st.sidebar.write("")

st.sidebar.write("Classes:")

st.sidebar.write(
    """
    ⚪ WBC  
    🔴 RBC  
    🟣 Platelets
    """
)


# ============================================================
# MODEL CLASSES
# ============================================================

st.sidebar.header("Model Classes")

for class_id, class_name in model.names.items():

    st.sidebar.write(
        f"{class_id} → {class_name}"
    )


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("📷 Upload Original Blood Smear Image")

uploaded_file = st.file_uploader(
    "Choose a microscope image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # OPEN ORIGINAL IMAGE
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # DISPLAY ORIGINAL IMAGE
    # --------------------------------------------------------

    st.subheader("📷 Original Image")

    st.image(
        image,
        caption="Uploaded microscope image",
        use_container_width=True
    )


    # --------------------------------------------------------
    # SAVE TEMPORARY IMAGE
    # --------------------------------------------------------

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False
        ) as temp_file:

            image.save(
                temp_file.name
            )

            temp_path = temp_file.name


        # ====================================================
        # RUN YOLO
        # ====================================================

        with st.spinner(
                "🔬 YOLO26 is analyzing the blood smear..."
        ):

            results = model.predict(
                source=temp_path,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                verbose=False
            )


        result = results[0]


        # ====================================================
        # COUNT CELLS
        # ====================================================

        counts = {
            "WBC": 0,
            "RBC": 0,
            "Platelets": 0
        }


        # ----------------------------------------------------
        # MODEL CLASS NAMES
        # ----------------------------------------------------

        names = model.names


        # ----------------------------------------------------
        # COUNT EACH DETECTION
        # ----------------------------------------------------

        if (
                result.boxes is not None
                and len(result.boxes) > 0
        ):

            for i in range(
                    len(result.boxes)
            ):

                class_id = int(
                    result.boxes.cls[i]
                )

                confidence = float(
                    result.boxes.conf[i]
                )

                class_name = names[
                    class_id
                ]


                if class_name in counts:

                    counts[
                        class_name
                    ] += 1


        # ====================================================
        # TOTAL CELL COUNT
        # ====================================================

        total_cells = (
                counts["WBC"]
                + counts["RBC"]
                + counts["Platelets"]
        )


        # ====================================================
        # CREATE TAGGED IMAGE
        # ====================================================

        annotated_image = result.plot()


        # ----------------------------------------------------
        # YOLO returns BGR.
        # Streamlit expects RGB.
        # ----------------------------------------------------

        annotated_image = (
            annotated_image[:, :, ::-1]
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.divider()

        st.subheader("🔬 AI Detection")


        image_column, results_column = st.columns(
            [2, 1]
        )


        # ====================================================
        # TAGGED IMAGE
        # ====================================================

        with image_column:

            st.markdown(
                "### 🏷️ AI Tagged Image"
            )

            st.image(
                annotated_image,
                caption="YOLO26 detected cells",
                use_container_width=True
            )


        # ====================================================
        # CELL COUNTS
        # ====================================================

        with results_column:

            st.markdown(
                "### 📊 Predicted Cell Count"
            )


            st.metric(
                "⚪ WBC",
                counts["WBC"]
            )


            st.metric(
                "🔴 RBC",
                counts["RBC"]
            )


            st.metric(
                "🟣 Platelets",
                counts["Platelets"]
            )


            st.divider()


            st.metric(
                "🩸 Total Cells",
                total_cells
            )


            st.divider()


            st.markdown(
                "### 🤖 Prediction Settings"
            )


            st.write(
                f"Confidence: {CONFIDENCE}"
            )


            st.write(
                f"Image size: {IMAGE_SIZE}"
            )


        # ====================================================
        # DETECTION SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Detection Summary"
        )


        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(
            4
        )


        with summary_col1:

            st.metric(
                "WBC",
                counts["WBC"]
            )


        with summary_col2:

            st.metric(
                "RBC",
                counts["RBC"]
            )


        with summary_col3:

            st.metric(
                "Platelets",
                counts["Platelets"]
            )


        with summary_col4:

            st.metric(
                "Total",
                total_cells
            )


        # ====================================================
        # MANUAL VERIFICATION
        # ====================================================

        st.divider()

        st.subheader(
            "👁️ Manual Verification"
        )


        st.info(
            """
            The boxes drawn on the image represent the cells
            detected by YOLO26.

            Check the tagged image to visually verify:

            • Whether each detected cell is real  
            • Whether the cell type is correct  
            • Whether any cells were missed  
            • Whether multiple cells were incorrectly grouped
            """
        )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            f"""
            Analysis complete!

            YOLO26 detected **{total_cells} cells**:
            **{counts["WBC"]} WBC + {counts["RBC"]} RBC + {counts["Platelets"]} Platelets**
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

                os.remove(
                    temp_path
                )

            except Exception:

                pass