"""vLLM inference engine."""

from typing import Any

from .base import BaseEngine


class VLLMEngine(BaseEngine):
    """Inference engine using vLLM."""
    
    def __init__(
        self,
        model_id: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        enforce_eager: bool = False,
        quantization: str | None = None,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.quantization = quantization
        self._llm = None
    
    def _load_model(self):
        """Lazy-load vLLM engine."""
        if self._llm is not None:
            return
        
        from vllm import LLM
        
        self._llm = LLM(
            model=self.model_id,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
            quantization=self.quantization,
        )
    
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate using vLLM."""
        self._load_model()
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        outputs = self._llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def get_engine_info(self) -> dict[str, Any]:
        """Return engine metadata."""
        import vllm
        
        return {
            "engine": "vllm",
            "version": vllm.__version__,
            "model_id": self.model_id,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "quantization": self.quantization,
        }
