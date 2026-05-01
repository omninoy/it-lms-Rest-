from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import HealthResponse, PredictResponse
from app.service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.prediction_service = PredictionService()
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to start service because model loading failed. "
            "Run train_model.py first."
        ) from exc
    yield


app = FastAPI(
    title="CV REST Service - Image Classification",
    description="FastAPI service for digit image classification.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    service: PredictionService = app.state.prediction_service
    try:
        content = await service.validate_and_read(file)
        prediction = service.predict(content)
        return PredictResponse.model_validate(prediction)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}",
        ) from exc
