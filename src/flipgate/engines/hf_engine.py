"""HuggingFace generate engine."""

from typing import Any

from .base import BaseEngine


class HFGenerateEngine(BaseEngine):
    """Inference engine using HuggingFace Transformers generate()."""
    
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_model_len: int = 4096,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.max_model_len = max_model_len
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        self._model.eval()
    
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate using HF model.generate()."""
        self._load_model()
        import torch
        
        bs = batch_size or 1
        all_responses = []
        
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_model_len - max_tokens,
            ).to(self._model.device)
            
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)
            
            # Decode only the generated portion
            input_lengths = inputs["input_ids"].shape[1]
            for j, output in enumerate(outputs):
                generated_ids = output[input_lengths:]
                response = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                all_responses.append(response)
        
        return all_responses
    
    def get_engine_info(self) -> dict[str, Any]:
        """Return engine metadata."""
        import transformers
        import torch
        
        return {
            "engine": "hf_generate",
            "version": transformers.__version__,
            "torch_version": torch.__version__,
            "model_id": self.model_id,
            "dtype": self.dtype,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
