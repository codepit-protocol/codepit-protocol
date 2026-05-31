"""Managed brain provider — POSTs to the engine's brain endpoint.

This is the V1 active provider. The engine handles tier resolution,
metering, and provider routing on its side; this client just signs a
bearer-authed POST and returns the ``content`` field from the response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx


_DEFAULT_TIMEOUT = 60.0


@dataclass(frozen=True)
class ManagedBrainResponse:
    """Full /v2/brain/generate response surfaced to callers that need the
    upstream provider + model attribution (e.g. for ``agent_decisions`` rows).

    ``content`` is the LLM text. ``provider`` / ``model`` are what the engine
    actually routed to after tier resolution — ``None`` only when the engine
    omitted them, which should not happen on the live path but is allowed by
    the schema so older engines remain compatible.
    """

    content: str
    provider: str | None = None
    model: str | None = None
    tier: str | None = None


class ManagedBrainError(RuntimeError):
    """Raised when the managed brain endpoint returns a non-2xx response."""

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class ManagedBrainProvider:
    """HTTP client for ``POST /v2/brain/generate``.

    The engine response shape is:
      {
        "content": "...",
        "tokens_in": 123,
        "tokens_out": 45,
        "latency_ms": 78,
        "tier": "cheap",
        "provider": "groq",
        "model": "...",
        "cost_codepit": "0",
        "cost_usd_micro": "180",
        "metering_enabled": false,
        "meter_status": "applied"
      }

    We surface ``content`` to the Brain. Token + cost telemetry stays on
    the engine ledger; the Python kit doesn't reimplement accounting.
    """

    def __init__(
        self,
        *,
        base_url: str,
        bearer_token: str,
        timeout_s: float = _DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("ManagedBrainProvider requires a base_url")
        if not bearer_token:
            raise ValueError("ManagedBrainProvider requires a bearer_token")
        self._base_url = base_url.rstrip("/")
        self._bearer = bearer_token
        self._timeout = timeout_s
        self._client = client

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
        """Return the LLM ``content`` string.

        Kept for callers that don't need provider attribution. New code
        should prefer :meth:`generate_with_metadata` so the resulting
        decision row can record which provider/model actually answered.
        """

        return self.generate_with_metadata(
            prompt=prompt,
            action_id=action_id,
            attempt=attempt,
            tier=tier,
            schema=schema,
            system=system,
        ).content

    def generate_with_metadata(
        self,
        *,
        prompt: str,
        action_id: str,
        attempt: int,
        tier: str,
        schema: Mapping[str, Any] | None = None,
        system: str | None = None,
    ) -> ManagedBrainResponse:
        """Call the engine brain gateway and return content + provider attribution."""

        body: dict[str, Any] = {
            "action_id": action_id,
            "attempt": attempt,
            "tier": tier,
            "prompt": prompt,
        }
        if system is not None:
            body["system"] = system
        if schema is not None:
            body["schema"] = dict(schema)

        if self._client is not None:
            response = self._client.post(
                f"{self._base_url}/v2/brain/generate",
                headers=self._headers(),
                json=body,
                timeout=self._timeout,
            )
        else:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/v2/brain/generate",
                    headers=self._headers(),
                    json=body,
                )

        if response.status_code >= 400:
            self._raise_from_error(response)

        try:
            payload = response.json()
        except ValueError as error:
            raise ManagedBrainError(
                f"managed brain returned non-JSON body: {error}",
                status_code=response.status_code,
            ) from error
        content = payload.get("content")
        if not isinstance(content, str):
            raise ManagedBrainError(
                "managed brain response missing 'content' string",
                status_code=response.status_code,
            )
        return ManagedBrainResponse(
            content=content,
            provider=_optional_str(payload.get("provider")),
            model=_optional_str(payload.get("model")),
            tier=_optional_str(payload.get("tier")),
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._bearer}",
            "content-type": "application/json",
        }

    def _raise_from_error(self, response: httpx.Response) -> None:
        message = f"managed brain HTTP {response.status_code}"
        code: str | None = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                code_value = error.get("code")
                if isinstance(code_value, str):
                    code = code_value
                msg_value = error.get("message")
                if isinstance(msg_value, str):
                    message = f"{message}: {msg_value}"
        raise ManagedBrainError(
            message,
            status_code=response.status_code,
            code=code,
        )


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["ManagedBrainError", "ManagedBrainProvider", "ManagedBrainResponse"]
