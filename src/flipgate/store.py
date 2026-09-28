"""Per-item JSONL results store with append-only semantics."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class ResultsStore:
    """Append-only JSONL store for per-item eval results."""
    
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_run_dir(self, run_id: str) -> Path:
        """Get directory for a specific run."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(exist_ok=True)
        return run_dir
    
    def append_item(
        self,
        run_id: str,
        dataset: str,
        item_id: str,
        prompt: str,
        response: str,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a single item result to the run's JSONL file."""
        run_dir = self._get_run_dir(run_id)
        filepath = run_dir / f"{dataset}.jsonl"
        
        record = {
            "item_id": item_id,
            "prompt": prompt,
            "response": response,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            record["metadata"] = metadata
        
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def append_run_metadata(
        self,
        run_id: str,
        model: str,
        engine: str,
        engine_version: str,
        dataset: str,
        manifest_sha: str,
        batch_size: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write run metadata file."""
        run_dir = self._get_run_dir(run_id)
        meta = {
            "run_id": run_id,
            "model": model,
            "engine": engine,
            "engine_version": engine_version,
            "dataset": dataset,
            "manifest_sha": manifest_sha,
            "batch_size": batch_size,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            meta.update(extra)
        
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    def iter_items(self, run_id: str, dataset: str) -> Iterator[dict[str, Any]]:
        """Iterate over all items in a run's dataset file."""
        filepath = self._get_run_dir(run_id) / f"{dataset}.jsonl"
        if not filepath.exists():
            return
        
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    
    def load_run_metadata(self, run_id: str) -> dict[str, Any]:
        """Load run metadata."""
        filepath = self._get_run_dir(run_id) / "metadata.json"
        if not filepath.exists():
            raise FileNotFoundError(f"No metadata for run {run_id}")
        with open(filepath) as f:
            return json.load(f)
    
    def list_runs(self) -> list[str]:
        """List all run IDs."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def get_items_df(self, run_id: str, dataset: str) -> list[dict[str, Any]]:
        """Load all items as a list of dicts (for analysis)."""
        return list(self.iter_items(run_id, dataset))


def generate_run_id(model: str, engine: str, dataset: str, batch_size: int | None = None) -> str:
    """Generate a deterministic run ID from parameters."""
    import time
    parts = [model, engine, dataset]
    if batch_size:
        parts.append(f"bs{batch_size}")
    # Use microsecond precision for uniqueness
    parts.append(datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"))
    # Add a random component to ensure uniqueness even within the same microsecond
    parts.append(str(time.time_ns()))
    raw = "_".join(parts)
    # Short hash for uniqueness
    hash_suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]
    batch_str = f"_bs{batch_size}" if batch_size else ""
    return f"{model.replace('/', '-')}_{engine}_{dataset}{batch_str}_{hash_suffix}"


def compute_manifest_sha(manifest_path: Path) -> str:
    """Compute SHA256 of manifest file for reproducibility tracking."""
    with open(manifest_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
