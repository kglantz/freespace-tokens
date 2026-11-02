"""Load configs/default.yaml, optionally overridden by configs/local.yaml (git-ignored)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_path(path: str | Path) -> Path:
    """Relative paths in configs are relative to the repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path | None = None) -> dict:
    """Default config merged with configs/local.yaml, or with `path` if given."""
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "default.yaml"))
    override = Path(path) if path else REPO_ROOT / "configs" / "local.yaml"
    if override.exists() and override.resolve() != (REPO_ROOT / "configs" / "default.yaml").resolve():
        cfg = _merge(cfg, yaml.safe_load(open(override)) or {})
    return cfg
