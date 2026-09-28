"""IFEval scorer: rule-based instruction following checks.

This is a simplified implementation. For production, vendor the official
google/IFEval checkers and wrap them here.
"""

import re
from typing import Any

from .base import BaseScorer


class IFEvalScorer(BaseScorer):
    """Score IFEval responses using rule-based checks.
    
    IFEval tests instruction following with various constraints:
    - Keyword inclusion/exclusion
    - Response length constraints
    - Format requirements (JSON, bullet points, etc.)
    - Language constraints
    
    This is a simplified version. For full accuracy, vendor the official
    google/IFEval checkers from https://github.com/google-research/IFEval
    """
    
    def score(self, prompt: str, response: str, reference: dict[str, Any]) -> float:
        """Score based on instruction following constraints."""
        instructions = reference.get("instruction_id_list", [])
        
        if not instructions:
            # No specific instructions to check
            return 1.0
        
        checks_passed = 0
        total_checks = len(instructions)
        
        for instruction_id in instructions:
            if self._check_instruction(instruction_id, prompt, response, reference):
                checks_passed += 1
        
        return checks_passed / total_checks if total_checks > 0 else 1.0
    
    def is_correct(self, prompt: str, response: str, reference: dict[str, Any]) -> bool:
        """Check if all instruction constraints are satisfied."""
        return self.score(prompt, response, reference) == 1.0
    
    def _check_instruction(
        self, instruction_id: str, prompt: str, response: str, reference: dict[str, Any]
    ) -> bool:
        """Check a single instruction constraint.
        
        This is a simplified implementation. The official IFEval has many more
        instruction types with specific checkers.
        """
        # Simplified checks for common instruction types
        if "keywords" in instruction_id.lower():
            return self._check_keywords(response, reference)
        elif "length" in instruction_id.lower() or "words" in instruction_id.lower():
            return self._check_length(response, reference)
        elif "format" in instruction_id.lower() or "json" in instruction_id.lower():
            return self._check_format(response, reference)
        else:
            # Unknown instruction type - assume pass
            # In production, this should vendor the official checker
            return True
    
    def _check_keywords(self, response: str, reference: dict[str, Any]) -> bool:
        """Check if required keywords are present."""
        required = reference.get("keywords", [])
        if not required:
            return True
        response_lower = response.lower()
        return all(kw.lower() in response_lower for kw in required)
    
    def _check_length(self, response: str, reference: dict[str, Any]) -> bool:
        """Check response length constraints."""
        min_words = reference.get("min_words")
        max_words = reference.get("max_words")
        
        word_count = len(response.split())
        
        if min_words and word_count < min_words:
            return False
        if max_words and word_count > max_words:
            return False
        
        return True
    
    def _check_format(self, response: str, reference: dict[str, Any]) -> bool:
        """Check format requirements (JSON, bullet points, etc.)."""
        required_format = reference.get("format", "").lower()
        
        if "json" in required_format:
            # Basic JSON check
            return response.strip().startswith("{") or response.strip().startswith("[")
        elif "bullet" in required_format:
            # Check for bullet points
            return bool(re.search(r"^[\s]*[-*•]", response, re.MULTILINE))
        
        return True
