"""Tests for results store."""

import pytest
import json
from pathlib import Path

from flipgate.store import ResultsStore, generate_run_id, compute_manifest_sha


@pytest.fixture
def store(tmp_path):
    return ResultsStore(tmp_path / "results")


class TestResultsStore:
    def test_append_and_read_item(self, store):
        store.append_item(
            run_id="test_run",
            dataset="gsm8k",
            item_id="item_1",
            prompt="What is 2+2?",
            response="4",
            score=1.0,
        )
        
        items = list(store.iter_items("test_run", "gsm8k"))
        assert len(items) == 1
        assert items[0]["item_id"] == "item_1"
        assert items[0]["score"] == 1.0
    
    def test_append_multiple_items(self, store):
        for i in range(3):
            store.append_item(
                run_id="test_run",
                dataset="gsm8k",
                item_id=f"item_{i}",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                score=float(i),
            )
        
        items = list(store.iter_items("test_run", "gsm8k"))
        assert len(items) == 3
    
    def test_append_with_metadata(self, store):
        store.append_item(
            run_id="test_run",
            dataset="gsm8k",
            item_id="item_1",
            prompt="Q",
            response="A",
            score=1.0,
            metadata={"batch_size": 8, "engine": "vllm"},
        )
        
        items = list(store.iter_items("test_run", "gsm8k"))
        assert items[0]["metadata"]["batch_size"] == 8
    
    def test_append_run_metadata(self, store):
        store.append_run_metadata(
            run_id="test_run",
            model="Qwen/Qwen2.5-3B-Instruct",
            engine="vllm",
            engine_version="0.2.7",
            dataset="gsm8k",
            manifest_sha="abc123",
            batch_size=8,
        )
        
        meta = store.load_run_metadata("test_run")
        assert meta["model"] == "Qwen/Qwen2.5-3B-Instruct"
        assert meta["batch_size"] == 8
    
    def test_iter_nonexistent_dataset(self, store):
        items = list(store.iter_items("test_run", "nonexistent"))
        assert items == []
    
    def test_load_nonexistent_metadata(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_run_metadata("nonexistent")
    
    def test_list_runs(self, store):
        store.append_item("run1", "gsm8k", "i1", "p", "r", 1.0)
        store.append_item("run2", "gsm8k", "i1", "p", "r", 1.0)
        
        runs = store.list_runs()
        assert set(runs) == {"run1", "run2"}
    
    def test_get_items_df(self, store):
        for i in range(5):
            store.append_item("run1", "gsm8k", f"i{i}", "p", "r", float(i))
        
        items = store.get_items_df("run1", "gsm8k")
        assert len(items) == 5


class TestRunIdGeneration:
    def test_generate_run_id_basic(self):
        run_id = generate_run_id("Qwen/Qwen2.5-3B", "vllm", "gsm8k")
        assert "Qwen-Qwen2.5-3B" in run_id
        assert "vllm" in run_id
        assert "gsm8k" in run_id
    
    def test_generate_run_id_with_batch_size(self):
        run_id = generate_run_id("model", "vllm", "gsm8k", batch_size=8)
        assert "bs8" in run_id
    
    def test_generate_run_id_unique(self):
        import time
        id1 = generate_run_id("model", "vllm", "gsm8k")
        time.sleep(0.01)
        id2 = generate_run_id("model", "vllm", "gsm8k")
        assert id1 != id2


class TestManifestSha:
    def test_compute_manifest_sha(self, tmp_path):
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("test content")
        
        sha = compute_manifest_sha(manifest_path)
        assert len(sha) == 64  # SHA256 hex length
        assert sha.isalnum()
