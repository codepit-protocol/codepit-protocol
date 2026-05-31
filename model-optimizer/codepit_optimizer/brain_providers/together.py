"""Together AI BYOK provider — V1 stub.

The V2 BYOK seam (Phase B) will let agent owners plug their own Together
API key into a ``together``-backed Brain that bypasses the engine's brain
endpoint entirely. Until then, the symbol exists so config files that
reference ``provider_name="together"`` don't break.
"""

from __future__ import annotations

from typing import Any, Mapping


class TogetherBrainProvider:
    """Phase B BYOK; not active in V1."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key

    def generate(
        self,
        *,
        prompt: str,
        action_id: str,
        attempt: int,
        tier: str,
        schema: Mapping[str, Any] | None = None,
        system: str | None = None,
    ) -> str:
        raise NotImplementedError(
            "BYOK provider; V2 BYOK ships in Phase B",
        )


__all__ = ["TogetherBrainProvider"]
