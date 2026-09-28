"""Base inference engine interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Abstract base class for inference engines."""
    
    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate responses for a batch of prompts.
        
        Args:
            prompts: List of formatted prompts (with chat template applied)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            batch_size: Override default batch size
            
        Returns:
            List of generated responses
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> dict[str, Any]:
        """Return engine metadata for reproducibility tracking."""
        pass
