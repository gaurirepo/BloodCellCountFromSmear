import streamlit as st

from ultralytics import YOLO

from PIL import Image

from collections import Counter

import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Blood Cell Smear Tracker",

    page_icon="🔬",

    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🔬 Real-Time Microscopic Smear Tracker"
)

st.write(
    "AI-powered detection and counting of "
    "RBCs, WBCs and platelets."
)


st.warning(
    "Research prototype only — this tool is "
    "not a medical diagnostic device."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(
        "runs/detect/"
        "blood_cell_detector/"
        "weights/model.pt"
    )


model = load_model()


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a microscope blood-smear image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.subheader(
        "Original Microscope Image"
    )


    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
            "🔬 Analyze Blood Smear",
            type="primary"
    ):


        with st.spinner(
                "AI is analyzing the blood smear..."
        ):


            results = model.predict(

                source=np.array(image),

                conf=0.50,

                verbose=False
            )


        result = results[0]


        # ====================================================
        # ANNOTATED IMAGE
        # ====================================================

        annotated_image = result.plot()


        st.subheader(
            "Detected Cells"
        )


        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )


        # ====================================================
        # COUNT CELLS
        # ====================================================

        counts = Counter()


        if result.boxes is not None:

            for cls in result.boxes.cls:

                class_id = int(
                    cls
                )

                class_name = model.names[
                    class_id
                ]

                counts[
                    class_name
                ] += 1


        # ====================================================
        # DISPLAY COUNTS
        # ====================================================

        st.subheader(
            "🩸 Cell Count"
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        col1.metric(

            "RBC",

            counts.get(
                "RBC",
                0
            )
        )


        col2.metric(

            "WBC",

            counts.get(
                "WBC",
                0
            )
        )


        col3.metric(

            "Platelets",

            counts.get(
                "Platelets",
                0
            )
        )


        total = sum(
            counts.values()
        )


        col4.metric(

            "Total Cells",

            total
        )


        # ====================================================
        # DETAILS
        # ====================================================

        st.subheader(
            "Detection Summary"
        )


        st.write(
            f"Total detected cells: **{total}**"
        )


        st.write(
            f"RBC: **{counts.get('RBC', 0)}**"
        )


        st.write(
            f"WBC: **{counts.get('WBC', 0)}**"
        )


        st.write(
            f"Platelets: **"
            f"{counts.get('Platelets', 0)}**"
        )


        # ====================================================
        # CONFIDENCE INFORMATION
        # ====================================================

        if result.boxes is not None:

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )


            if len(confidences) > 0:

                average_confidence = (
                    float(
                        np.mean(
                            confidences
                        )
                    )
                )


                st.metric(

                    "Average Detection Confidence",

                    f"{average_confidence:.1%}"
                )