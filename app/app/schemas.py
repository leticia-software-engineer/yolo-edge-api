from typing import List, Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    model_name: str = "yolov8n.pt"
    confidence: float = 0.25


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]


class PredictResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float
    model_used: str
    image_width: int
    image_height: int


class BatchPredictRequest(BaseModel):
    images_base64: List[str]
    model_name: str = "yolov8n.pt"
    confidence: float = 0.25


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    avg_inference_ms: float
