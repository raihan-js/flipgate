"""Frozen eval manifest loader and validator."""

from pathlib import Path
from typing import Any

import yaml


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    """Load the frozen eval manifest."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "manifest.yaml"
    else:
        path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    
    with open(path) as f:
        manifest = yaml.safe_load(f)
    
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate manifest structure and required fields."""
    required_sections = ["manifest_version", "models", "sampling", "engines", "datasets"]
    for section in required_sections:
        if section not in manifest:
            raise ValueError(f"Manifest missing required section: {section}")
    
    # Validate sampling params enforce greedy decoding
    sampling = manifest["sampling"]
    if sampling.get("temperature", 0) != 0.0:
        raise ValueError("FlipGate requires temperature=0.0 for deterministic comparison")
    
    # Validate datasets have required fields
    for name, ds in manifest["datasets"].items():
        required_ds_fields = ["source", "scorer", "num_items"]
        for field in required_ds_fields:
            if field not in ds:
                raise ValueError(f"Dataset {name} missing required field: {field}")


def get_model_config(manifest: dict[str, Any], model_key: str) -> dict[str, Any]:
    """Get model configuration by key."""
    if model_key not in manifest["models"]:
        raise KeyError(f"Model {model_key} not found in manifest")
    return manifest["models"][model_key]


def get_dataset_config(manifest: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    """Get dataset configuration by key."""
    if dataset_key not in manifest["datasets"]:
        raise KeyError(f"Dataset {dataset_key} not found in manifest")
    return manifest["datasets"][dataset_key]
