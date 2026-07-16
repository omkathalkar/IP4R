"""Config loading. Thresholds and paths come from YAML, never hard-coded in logic."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = three parents up from this file (src/ip4r/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


class Config:
    """Thin wrapper over the parsed YAML dict with dotted-path access."""

    def __init__(self, data: dict[str, Any], root: Path = REPO_ROOT):
        self._data = data
        self.root = root

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a config path value relative to the repo root."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"Missing path config: {dotted}")
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def data(self) -> dict[str, Any]:
        return self._data
