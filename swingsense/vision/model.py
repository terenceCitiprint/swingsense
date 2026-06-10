"""Fetch and cache the MediaPipe pose-landmarker model."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .. import config

_MODELS = {
    "lite": "pose_landmarker_lite",
    "full": "pose_landmarker_full",
    "heavy": "pose_landmarker_heavy",
}
_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "{name}/float16/latest/{name}.task"
)


def ensure_model(variant: str = "full") -> Path:
    """Return a local path to the model, downloading it once if needed."""
    if variant not in _MODELS:
        raise ValueError(f"Unknown model variant {variant!r}; pick {list(_MODELS)}")
    name = _MODELS[variant]
    models_dir = config.data_home() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / f"{name}.task"
    if not dest.exists():
        url = _URL.format(name=name)
        urllib.request.urlretrieve(url, dest)
    return dest
