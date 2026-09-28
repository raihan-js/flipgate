"""Tests for manifest loader."""

import pytest
import tempfile
from pathlib import Path
import yaml

from flipgate.manifest import load_manifest, get_model_config, get_dataset_config


@pytest.fixture
def valid_manifest():
    return {
        "manifest_version": "1.0.0",
        "models": {
            "base": {
                "id": "Qwen/Qwen2.5-3B-Instruct",
                "dtype": "bfloat16",
            }
        },
        "sampling": {
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "engines": {
            "vllm": {
                "version": "0.2.7",
            }
        },
        "datasets": {
            "gsm8k": {
                "source": "openai/gsm8k",
                "scorer": "gsm8k",
                "num_items": 1000,
            }
        },
    }


@pytest.fixture
def manifest_file(valid_manifest, tmp_path):
    path = tmp_path / "manifest.yaml"
    with open(path, "w") as f:
        yaml.dump(valid_manifest, f)
    return path


class TestManifest:
    def test_load_valid_manifest(self, manifest_file):
        manifest = load_manifest(manifest_file)
        assert manifest["manifest_version"] == "1.0.0"
        assert "base" in manifest["models"]
    
    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/manifest.yaml")
    
    def test_validate_missing_section(self, tmp_path):
        manifest = {"manifest_version": "1.0.0"}  # missing required sections
        path = tmp_path / "manifest.yaml"
        with open(path, "w") as f:
            yaml.dump(manifest, f)
        
        with pytest.raises(ValueError, match="missing required section"):
            load_manifest(path)
    
    def test_validate_nonzero_temperature(self, tmp_path):
        manifest = {
            "manifest_version": "1.0.0",
            "models": {},
            "sampling": {"temperature": 0.5},  # must be 0.0
            "engines": {},
            "datasets": {},
        }
        path = tmp_path / "manifest.yaml"
        with open(path, "w") as f:
            yaml.dump(manifest, f)
        
        with pytest.raises(ValueError, match="temperature=0.0"):
            load_manifest(path)
    
    def test_validate_dataset_missing_field(self, tmp_path):
        manifest = {
            "manifest_version": "1.0.0",
            "models": {},
            "sampling": {"temperature": 0.0},
            "engines": {},
            "datasets": {
                "test": {
                    "source": "test/dataset"
                    # missing "scorer" and "num_items"
                }
            },
        }
        path = tmp_path / "manifest.yaml"
        with open(path, "w") as f:
            yaml.dump(manifest, f)
        
        with pytest.raises(ValueError, match="missing required field"):
            load_manifest(path)
    
    def test_get_model_config(self, valid_manifest):
        config = get_model_config(valid_manifest, "base")
        assert config["id"] == "Qwen/Qwen2.5-3B-Instruct"
    
    def test_get_model_config_missing(self, valid_manifest):
        with pytest.raises(KeyError):
            get_model_config(valid_manifest, "nonexistent")
    
    def test_get_dataset_config(self, valid_manifest):
        config = get_dataset_config(valid_manifest, "gsm8k")
        assert config["source"] == "openai/gsm8k"
    
    def test_get_dataset_config_missing(self, valid_manifest):
        with pytest.raises(KeyError):
            get_dataset_config(valid_manifest, "nonexistent")
