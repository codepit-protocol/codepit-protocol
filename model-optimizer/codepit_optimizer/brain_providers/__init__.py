"""Brain provider implementations.

V1 active: ``managed`` — POSTs to the engine's ``/v2/brain/generate``
endpoint using the agent's bearer token. This is the only path that flows
real LLM traffic in V1.

V1 stubs: ``groq``, ``openai``, ``together`` — BYOK paths that ship with
``NotImplementedError`` so the import surface is stable while the V2 BYOK
seam lands in Phase B. They exist purely so that flipping
``provider_name`` in a config file later is a config change, not a refactor.
"""

from .managed import ManagedBrainProvider

__all__ = ["ManagedBrainProvider"]
