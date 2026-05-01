from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class PredictResponse(BaseModel):
    class_name: str = Field(alias="class")
    probability: float
    all_classes: dict[str, float]

    model_config = {"populate_by_name": True}
