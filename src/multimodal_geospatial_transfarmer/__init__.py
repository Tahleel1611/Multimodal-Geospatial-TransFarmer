"""Core package for the multimodal geospatial UAV-IoT pipeline."""

from .fusion import BaselineFusionModel
from .preprocessing import align_modalities, interpolate_iot_samples
from .schemas import GeoTile, IoTSample, MultimodalSample, PredictionResult, UAVFrame

__all__ = [
    "GeoTile",
    "IoTSample",
    "MultimodalSample",
    "PredictionResult",
    "UAVFrame",
    "BaselineFusionModel",
    "align_modalities",
    "interpolate_iot_samples",
]