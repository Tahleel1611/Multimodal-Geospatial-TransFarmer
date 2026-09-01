"""Minimal YAML configuration loading with clear validation errors."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "configs/default_config.yaml") -> dict[str, Any]:
    """Load a YAML mapping; keeps configuration optional for lightweight installs."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to load YAML configuration files") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    return config
