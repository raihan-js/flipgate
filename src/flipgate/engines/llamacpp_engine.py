"""llama.cpp inference engine (GGUF)."""

from typing import Any

from .base import BaseEngine


class LlamaCppEngine(BaseEngine):
    """Inference engine using llama.cpp via llama-cpp-python."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        n_threads: int = 6,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self._llm = None
    
    def _load_model(self):
        """Lazy-load llama.cpp model."""
        if self._llm is not None:
            return
        
        from llama_cpp import Llama
        
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=False,
        )
    
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate using llama.cpp."""
        self._load_model()
        
        responses = []
        for prompt in prompts:
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            responses.append(output["choices"][0]["text"])
        
        return responses
    
    def get_engine_info(self) -> dict[str, Any]:
        """Return engine metadata."""
        return {
            "engine": "llama_cpp",
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
        }
