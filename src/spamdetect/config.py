"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config; relative ``paths`` entries are resolved against the config's folder."""
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base = path.parent
    cfg["paths"] = {k: str((base / v).resolve()) for k, v in cfg["paths"].items()}
    cfg["_config_path"] = str(path)
    return cfg
