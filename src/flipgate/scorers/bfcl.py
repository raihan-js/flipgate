"""BFCL (Berkeley Function Calling Leaderboard) scorer.

Evaluates function calling accuracy by checking if the model's output
matches the expected function calls.
"""

import json
import re
from typing import Any

from .base import BaseScorer


class BFCLScorer(BaseScorer):
    """Score function calling accuracy using BFCL format."""
    
    def score(self, prompt: str, response: str, reference: dict[str, Any]) -> float:
        """Score a function calling response.

        Two modes (both honest, documented per item):
        - spec mode: reference has "function" specs -> executable-call rate
          (right name + all required args). Used for simple/parallel sets.
        - gold mode: reference has "ground_truth" call strings ->
          exact match on name + args (numeric tolerance). Used for exec sets.

        Returns:
            Score between 0.0 and 1.0
        """
        if reference.get("ground_truth"):
            return self._score_exact(response, reference["ground_truth"])

        expected_calls = reference.get("function", [])
        if not expected_calls:
            return 1.0  # No expected calls, so response is correct

        # Parse function calls from response
        predicted_calls = self._extract_function_calls(response)

        if not predicted_calls:
            return 0.0  # No function calls found

        # Compare predicted vs expected
        return self._compare_calls(predicted_calls, expected_calls)

    @staticmethod
    def _parse_call_string(s: str) -> tuple[str, dict] | None:
        """Parse 'name(k=v, k2=v2)' into (name, args). Returns None on failure."""
        m = re.match(r"\s*([A-Za-z0-9_.]+)\s*\((.*)\)\s*$", s, re.DOTALL)
        if not m:
            return None
        name, argstr = m.group(1), m.group(2)
        args: dict[str, Any] = {}
        for part in argstr.split(","):
            if "=" not in part:
                return None
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip().strip("\"'")
        return name, args

    @staticmethod
    def _values_equal(a: str, b: str) -> bool:
        """Numeric-tolerant comparison (handles 1/6 vs 0.1667, '5' vs 5)."""
        def num(x: str):
            x = str(x).strip()
            try:
                if "/" in x and x.count("/") == 1:
                    p, q = x.split("/")
                    return float(p) / float(q)
                return float(x)
            except (ValueError, ZeroDivisionError):
                return None
        na, nb = num(a), num(b)
        if na is not None and nb is not None:
            return abs(na - nb) < 1e-6
        return str(a).strip().lower() == str(b).strip().lower()

    def _score_exact(self, response: str, ground_truth: list[str]) -> float:
        """Exact match against gold call strings. All gold calls must be
        matched by distinct predicted calls (name + every gold arg)."""
        predicted = self._extract_function_calls(response)
        if not predicted:
            return 0.0
        gold = [self._parse_call_string(g) for g in ground_truth]
        gold = [g for g in gold if g is not None]
        if not gold:
            return 0.0
        used = set()
        matched = 0
        for gname, gargs in gold:
            for i, p in enumerate(predicted):
                if i in used:
                    continue
                if p.get("name") != gname:
                    continue
                pargs = p.get("arguments", {})
                if all(k in pargs and self._values_equal(pargs[k], v)
                       for k, v in gargs.items()):
                    used.add(i)
                    matched += 1
                    break
        return matched / len(gold)
    
    def _extract_function_calls(self, response: str) -> list[dict]:
        """Extract function calls from model response.
        
        Looks for JSON objects with "name" and "arguments" keys.
        Handles nested braces properly.
        """
        calls = []
        
        # Find all JSON-like objects with proper brace matching
        i = 0
        while i < len(response):
            if response[i] == '{':
                # Find matching closing brace
                depth = 0
                for j in range(i, len(response)):
                    if response[j] == '{':
                        depth += 1
                    elif response[j] == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = response[i:j+1]
                            try:
                                obj = json.loads(json_str)
                                if "name" in obj and "arguments" in obj:
                                    calls.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
                else:
                    i += 1
            else:
                i += 1
        
        return calls
    
    def _compare_calls(self, predicted: list[dict], expected: list[dict]) -> float:
        """Compare predicted function calls against expected ones.
        
        Returns a score based on:
        - Correct function name
        - Correct arguments (name and value)
        """
        if not predicted or not expected:
            return 0.0
        
        total_score = 0.0
        num_expected = len(expected)
        
        for exp_call in expected:
            exp_name = exp_call.get("name", "")
            exp_args = exp_call.get("parameters", {}).get("properties", {})
            
            # Find matching predicted call
            best_match_score = 0.0
            for pred_call in predicted:
                pred_name = pred_call.get("name", "")
                pred_args = pred_call.get("arguments", {})
                
                # Check function name
                name_match = (pred_name == exp_name)
                
                # Check arguments
                arg_score = 0.0
                num_required_args = len(exp_args)
                if num_required_args > 0:
                    correct_args = 0
                    for arg_name in exp_args:
                        if arg_name in pred_args:
                            # For now, just check if argument exists
                            # Could be more strict about values
                            correct_args += 1
                    arg_score = correct_args / num_required_args
                else:
                    arg_score = 1.0
                
                # Combined score for this call
                if name_match:
                    call_score = 0.5 + 0.5 * arg_score  # Name is 50%, args are 50%
                else:
                    call_score = 0.0
                
                best_match_score = max(best_match_score, call_score)
            
            total_score += best_match_score
        
        return total_score / num_expected
    
    def is_correct(self, prompt: str, response: str, reference: dict[str, Any]) -> bool:
        """Check if response is correct (score == 1.0)."""
        return self.score(prompt, response, reference) == 1.0
