"""FastAPI inference service for SmearDx.

POST /analyze infers at min(class gates)=0.40 and returns every box at that
floor so the UI can apply live gates. Default counts and the annotated PNG
still use RBC ≥ 0.60, WBC ≥ 0.40, Platelets ≥ 0.40.
Canonical mapping: {0: Platelets, 1: RBC, 2: WBC}.
"""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from eval_config import (
    CANONICAL_CLASS_NAMES,
    CLASS_COLORS_RGB,
    CLASS_THRESHOLDS,
    IMAGE_SIZE,
    MIN_INFERENCE_CONF,
    accept_detection,
    assert_canonical_mapping,
    empty_class_counts,
    resolve_final_model,
)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
    "application/octet-stream",
}

BOX_COLORS = {
    "RBC": CLASS_COLORS_RGB["RBC"],
    "WBC": CLASS_COLORS_RGB["WBC"],
    "Platelets": CLASS_COLORS_RGB["Platelets"],
}


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


def annotate_image(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    label_font = _font(14)

    for detection in detections:
        class_name = detection["class"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["box"]
        color = BOX_COLORS.get(class_name, (148, 163, 184))

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


def image_to_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_model() -> YOLO:
    model_path = resolve_final_model()
    model = YOLO(str(model_path))
    assert_canonical_mapping(model.names)
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.model = load_model()
    yield


app = FastAPI(
    title="SmearDx Inference API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    model: YOLO = app.state.model

    try:
        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=MIN_INFERENCE_CONF,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    result = results[0]
    raw: list[dict[str, Any]] = []

    if result.boxes is not None and len(result.boxes) > 0:
        for index in range(len(result.boxes)):
            class_id = int(result.boxes.cls[index])
            confidence = float(result.boxes.conf[index])
            class_name = CANONICAL_CLASS_NAMES.get(
                class_id,
                str(model.names[class_id]),
            )
            raw.append(
                {
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "box": [float(value) for value in result.boxes.xyxy[index].tolist()],
                }
            )

    accepted = [
        detection
        for detection in raw
        if accept_detection(detection["class"], detection["confidence"])
    ]
    counts = empty_class_counts()
    for detection in accepted:
        if detection["class"] in counts:
            counts[detection["class"]] += 1

    annotated = annotate_image(image, accepted)

    return {
        "counts": {
            "rbc": counts["RBC"],
            "wbc": counts["WBC"],
            "platelets": counts["Platelets"],
            "total": sum(counts.values()),
        },
        "boxes": raw,
        "annotated_image_base64": image_to_png_base64(annotated),
        "operating_point": {
            "rbc": CLASS_THRESHOLDS["RBC"],
            "wbc": CLASS_THRESHOLDS["WBC"],
            "platelets": CLASS_THRESHOLDS["Platelets"],
            "infer_conf": MIN_INFERENCE_CONF,
            "mapping": CANONICAL_CLASS_NAMES,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
