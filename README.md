# Multimodel Geospatial Transfarmer 

This project delves into a comprehensive analysis using Satellite Imagery, UAV Data and IOT signal sensors in order to account for drastic changes in weather and climate. 

## Runnable hybrid demo

`run_hybrid_pest_density_demo.py` trains an end-to-end PyTorch model that fuses
UAV/crop imagery with a 24-hour temperature, relative-humidity, soil-moisture,
and trap-count stream to regress field pest density. It uses a real local
PlantVillage/IP102-style `ImageFolder` dataset when supplied, optionally tries
CIFAR-10, and defaults to deterministic procedural UAV-like imagery so it runs
offline:

```powershell
pip install -e .
python run_hybrid_pest_density_demo.py
python run_hybrid_pest_density_demo.py --image-root data/PlantVillage --epochs 3
python run_hybrid_pest_density_demo.py --download-cifar
```

It prints training and MAE/RMSE/R² evaluation metrics, simulates INT8 dynamic
quantized edge inference, reports latency, and writes a Grad-CAM heatmap to
`outputs/gradcam_attribution.png`.

## Modular production layout

The refactored implementation is organized into `src/data`, `src/models`,
`src/utils`, and `src/export`, with YAML defaults in `configs/` and CLI entry
points in `scripts/`. Install the package and run:

```powershell
python scripts/train.py --use-synthetic-telemetry --epochs 3 --export-onnx
python scripts/evaluate.py --checkpoint outputs/model.pt --xai-enabled
python scripts/export_onnx.py --checkpoint outputs/model.pt --validate
pytest -q
```

Hyperparameters and deployment flags are centralized in
`configs/default_config.yaml`; `src/utils/config.py` provides the validated
loader, while CLI flags make run-specific overrides explicit.

Set `--image-root` to an ImageFolder-formatted PlantVillage/IP102 directory and
provide `--telemetry-csv` for field telemetry. The CSV must contain
`temperature_c`, `relative_humidity`, and `soil_moisture`; optional columns are
`sample_id`, `timestamp`, `trap_count`, and `pest_density`.
