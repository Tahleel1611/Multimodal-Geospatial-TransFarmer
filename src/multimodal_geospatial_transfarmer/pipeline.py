from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .fusion import BaselineFusionModel
from .preprocessing import align_modalities
from .schemas import GeoTile, IoTSample, MultimodalSample, PredictionResult, UAVFrame


@dataclass(slots=True)
class MultimodalInferencePipeline:
    """Minimal end-to-end orchestration layer."""

    model: BaselineFusionModel

    def build_samples(
        self,
        uav_frames: Sequence[UAVFrame],
        iot_samples: Sequence[IoTSample],
        geo_tiles: Sequence[GeoTile],
    ) -> list[MultimodalSample]:
        return align_modalities(uav_frames=uav_frames, iot_samples=iot_samples, geo_tiles=geo_tiles)

    def predict_batch(
        self,
        uav_frames: Sequence[UAVFrame],
        iot_samples: Sequence[IoTSample],
        geo_tiles: Sequence[GeoTile],
    ) -> list[PredictionResult]:
        samples = self.build_samples(uav_frames=uav_frames, iot_samples=iot_samples, geo_tiles=geo_tiles)
        return [self.model.predict(sample) for sample in samples]