"""Minimal YAML configuration loader used by the training and evaluation tools."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(path: str | Path) -> SimpleNamespace:
    """Flatten the TRAIN, MODEL, and DATA YAML sections into an attribute object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    values: dict[str, object] = {}
    for section in ("TRAIN", "MODEL", "DATA"):
        values.update(raw.get(section, {}))
    return SimpleNamespace(**values)
