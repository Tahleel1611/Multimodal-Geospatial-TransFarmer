from __future__ import annotations

from dataclasses import dataclass
from math import exp
from statistics import mean

from .schemas import MultimodalSample, PredictionResult


@dataclass(slots=True)
class BaselineFusionModel:
    """Deterministic starter model for multimodal scoring.

    This is intentionally simple so the repository has a concrete,
    testable implementation before adding deep backbones.
    """

    pest_density_scale: float = 1.0
    disease_bias: float = 0.0

    def predict(self, sample: MultimodalSample) -> PredictionResult:
        image_signal = 0.0
        if sample.uav_frame.sensor_mode == "multispectral":
            image_signal += 0.6
        elif sample.uav_frame.sensor_mode == "rgb":
            image_signal += 0.3

        env_features = []
        feature_importance: dict[str, float] = {}

        if sample.iot_sample is not None:
            if sample.iot_sample.relative_humidity is not None:
                humidity_score = max(0.0, min(1.0, sample.iot_sample.relative_humidity / 100.0))
                env_features.append(humidity_score)
                feature_importance["relative_humidity"] = humidity_score
                image_signal += humidity_score * 0.2
            if sample.iot_sample.temperature_c is not None:
                temperature_score = max(0.0, min(1.0, abs(sample.iot_sample.temperature_c - 28.0) / 20.0))
                env_features.append(temperature_score)
                feature_importance["temperature_c"] = temperature_score
                image_signal += temperature_score * 0.1
            if sample.iot_sample.soil_moisture is not None:
                moisture_score = max(0.0, min(1.0, sample.iot_sample.soil_moisture / 100.0))
                env_features.append(moisture_score)
                feature_importance["soil_moisture"] = moisture_score
                image_signal += moisture_score * 0.15

        if sample.geo_tile is not None and sample.geo_tile.embedding:
            embedding_strength = mean(abs(value) for value in sample.geo_tile.embedding)
            feature_importance["geo_embedding_strength"] = embedding_strength
            image_signal += min(1.0, embedding_strength) * 0.2

        pest_density = max(0.0, self.pest_density_scale * image_signal * (1.0 + 0.1 * len(env_features)))
        disease_probability = 1.0 / (1.0 + exp(-(image_signal + self.disease_bias)))
        disease_label = "disease_present" if disease_probability >= 0.5 else "healthy"
        confidence = max(disease_probability, 1.0 - disease_probability)

        return PredictionResult(
            pest_density=pest_density,
            disease_probability=disease_probability,
            disease_label=disease_label,
            confidence=confidence,
            feature_importance=feature_importance,
        )