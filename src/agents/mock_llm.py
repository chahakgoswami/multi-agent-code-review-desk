"""MockLLMBackend: deterministic canned responses keyed by input hash."""
from __future__ import annotations

import hashlib
import textwrap
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Canned response catalogue
# Each entry maps a *partial* hash prefix (first 4 hex chars) to a response.
# For hashes that don't match any prefix we fall back to a generic template.
# ---------------------------------------------------------------------------

_CANNED: Dict[str, str] = {
    "0000": "The code looks clean. No obvious issues detected.",
    "0001": "Potential off-by-one error detected in loop bounds.",
    "0002": "Variable names are not descriptive enough.",
    "0003": "Function complexity is high; consider decomposition.",
    "0004": "Missing docstring on public function.",
    "0005": "Hardcoded credential string found.",
    "0006": "SQL query constructed via string concatenation.",
    "0007": "Use of eval() detected; potential code injection risk.",
    "0008": "Unsafe deserialization with pickle.loads detected.",
    "0009": "Exception silenced with bare except clause.",
    "000a": "Mutable default argument in function signature.",
    "000b": "Global variable mutation inside function.",
    "000c": "Unreachable code after return statement.",
    "000d": "Shadowing of built-in name detected.",
    "000e": "Missing type annotations on function parameters.",
    "000f": "Circular import risk identified.",
}

_GENERIC_TEMPLATE = textwrap.dedent(
    """\
    LLM analysis complete (hash={hash}).
    The submitted code contains {line_count} line(s).
    No specific pattern matched; general review applied.
    Recommendation: ensure adequate test coverage and documentation.
    """
)


class MockLLMBackend:
    """Simulate an LLM that returns deterministic responses based on input hash."""

    def __init__(self, seed: int = 42) -> None:
        """seed is not used for randomness but stored for reproducibility metadata."""
        self.seed = seed
        self._last_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        """Return a canned response for *prompt*.

        The response is chosen deterministically:
        1. SHA-256 hash of the prompt is computed.
        2. First 4 hex characters are looked up in the canned catalogue.
        3. If no match, a generic template is filled in.
        """
        h = self._hash(prompt)
        self._last_hash = h
        prefix = h[:4].lower()
        if prefix in _CANNED:
            return _CANNED[prefix]
        line_count = prompt.count("\n") + 1
        return _GENERIC_TEMPLATE.format(hash=h[:8], line_count=line_count)

    def last_hash(self) -> Optional[str]:
        """Return the SHA-256 hex digest of the most recent prompt, or None."""
        return self._last_hash

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:  # pragma: no cover
        return f"MockLLMBackend(seed={self.seed})"
