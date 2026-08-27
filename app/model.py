import torch
import ultralytics
from ultralytics import YOLO

# Permite que o PyTorch 2.6+ carregue as classes do Ultralytics com seguranca
try:
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
except AttributeError:
    pass

_models = {}

def get_default_model_name() -> str:
    return "yolov8n.pt"

def load_model(model_name: str = "yolov8n.pt"):
    if model_name not in _models:
        _models[model_name] = YOLO(model_name)
    return _models[model_name]
