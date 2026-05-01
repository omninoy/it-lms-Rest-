from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, UploadFile, status

from app.config import (
    ALLOWED_CONTENT_TYPES,
    CLASS_NAMES,
    MAX_FILE_SIZE_BYTES,
    MODEL_PATH,
)
from app.model import DigitCNN


class PredictionService:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> DigitCNN:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file not found: {MODEL_PATH}. "
                "Train and save the model first (run train_model.py)."
            )

        model = DigitCNN(num_classes=len(CLASS_NAMES))
        state = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        return model

    async def validate_and_read(self, file: UploadFile) -> bytes:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            allowed = ", ".join(sorted(ALLOWED_CONTENT_TYPES))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed}",
            )

        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file was uploaded.",
            )

        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File is too large. Max size is {MAX_FILE_SIZE_BYTES} bytes.",
            )

        return content

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("L")
        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid image.",
            ) from exc

        image = image.resize((32, 32))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_bytes: bytes) -> dict:
        x = self.preprocess_image(image_bytes)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        all_classes = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}
        return {
            "class": CLASS_NAMES[top_idx],
            "probability": float(probs[top_idx]),
            "all_classes": all_classes,
        }
