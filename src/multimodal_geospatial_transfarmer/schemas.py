from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class UAVFrame:
    """Single UAV capture with spatial and acquisition metadata."""

    image_path: str
    timestamp: datetime
    latitude: float
    longitude: float
    altitude_m: float
    sensor_mode: str = "rgb"
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IoTSample:
    """In-field telemetry aligned to a geographic point."""

    timestamp: datetime
    latitude: float
    longitude: float
    soil_moisture: float | None
    relative_humidity: float | None
    temperature_c: float | None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeoTile:
    """Satellite or foundation-model derived geospatial representation."""

    tile_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    embedding: Sequence[float]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MultimodalSample:
    """Joined sample used by training and inference pipelines."""

    uav_frame: UAVFrame
    iot_sample: IoTSample | None
    geo_tile: GeoTile | None
    target_pest_density: float | None = None
    target_disease_label: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Unified model output for downstream UI and analytics."""

    pest_density: float
    disease_probability: float
    disease_label: str
    confidence: float
    feature_importance: Mapping[str, float] = field(default_factory=dict)
    spatial_attention_map_path: str | None = None