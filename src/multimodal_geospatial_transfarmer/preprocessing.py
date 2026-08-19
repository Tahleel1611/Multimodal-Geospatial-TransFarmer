from __future__ import annotations

from datetime import datetime
from math import cos, hypot, radians
from typing import Sequence

from .schemas import GeoTile, IoTSample, MultimodalSample, UAVFrame


def _distance_meters(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Approximate local distance for alignment without external geodesy deps."""

    lat_scale = 111_320.0
    lon_scale = 111_320.0 * cos(radians((lat_a + lat_b) / 2.0))
    return hypot((lat_a - lat_b) * lat_scale, (lon_a - lon_b) * lon_scale)


def interpolate_iot_samples(
    samples: Sequence[IoTSample],
    target_time: datetime,
    max_gap_minutes: int = 30,
) -> IoTSample | None:
    """Return the nearest telemetry sample within a temporal tolerance."""

    if not samples:
        return None

    nearest = min(samples, key=lambda sample: abs((sample.timestamp - target_time).total_seconds()))
    if abs((nearest.timestamp - target_time).total_seconds()) > max_gap_minutes * 60:
        return None
    return nearest


def _nearest_geo_tile(
    tiles: Sequence[GeoTile],
    latitude: float,
    longitude: float,
    target_time: datetime,
    max_distance_m: float = 500.0,
    max_time_gap_minutes: int = 60,
) -> GeoTile | None:
    if not tiles:
        return None

    candidates = []
    for tile in tiles:
        time_gap = abs((tile.timestamp - target_time).total_seconds())
        if time_gap > max_time_gap_minutes * 60:
            continue
        distance = _distance_meters(latitude, longitude, tile.latitude, tile.longitude)
        if distance <= max_distance_m:
            candidates.append((distance, time_gap, tile))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def align_modalities(
    uav_frames: Sequence[UAVFrame],
    iot_samples: Sequence[IoTSample],
    geo_tiles: Sequence[GeoTile],
) -> list[MultimodalSample]:
    """Create aligned multimodal samples around each UAV frame."""

    aligned_samples: list[MultimodalSample] = []
    for frame in uav_frames:
        nearby_iot = [
            sample
            for sample in iot_samples
            if abs((sample.timestamp - frame.timestamp).total_seconds()) <= 30 * 60
            and _distance_meters(frame.latitude, frame.longitude, sample.latitude, sample.longitude) <= 1000.0
        ]
        iot_sample = interpolate_iot_samples(nearby_iot, frame.timestamp)
        geo_tile = _nearest_geo_tile(geo_tiles, frame.latitude, frame.longitude, frame.timestamp)
        aligned_samples.append(MultimodalSample(uav_frame=frame, iot_sample=iot_sample, geo_tile=geo_tile))
    return aligned_samples