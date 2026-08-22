from datetime import datetime

from multimodal_geospatial_transfarmer.fusion import BaselineFusionModel
from multimodal_geospatial_transfarmer.pipeline import MultimodalInferencePipeline
from multimodal_geospatial_transfarmer.schemas import GeoTile, IoTSample, UAVFrame


def test_pipeline_predicts_from_aligned_modalities() -> None:
    pipeline = MultimodalInferencePipeline(model=BaselineFusionModel())

    frame = UAVFrame(
        image_path="frame_001.png",
        timestamp=datetime(2026, 8, 15, 9, 0, 0),
        latitude=12.9716,
        longitude=77.5946,
        altitude_m=40.0,
        sensor_mode="multispectral",
    )
    iot_sample = IoTSample(
        timestamp=datetime(2026, 8, 15, 9, 2, 0),
        latitude=12.9717,
        longitude=77.5947,
        soil_moisture=35.0,
        relative_humidity=78.0,
        temperature_c=30.0,
    )
    geo_tile = GeoTile(
        tile_id="tile_001",
        timestamp=datetime(2026, 8, 15, 9, 0, 0),
        latitude=12.9716,
        longitude=77.5946,
        embedding=[0.1, -0.2, 0.3],
    )

    outputs = pipeline.predict_batch([frame], [iot_sample], [geo_tile])

    assert len(outputs) == 1
    assert outputs[0].pest_density >= 0.0
    assert outputs[0].disease_label in {"healthy", "disease_present"}