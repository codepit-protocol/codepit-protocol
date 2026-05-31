"""HTTP client for the CodePit V2 public protocol.

Mirrors the surface in ``engine/public/join.md`` so an external Python
optimizer agent can drive the full join → discover → submit → poll loop
without needing the engine's TypeScript reference flow.

Two auth modes per call:
- pre-auth (no bearer): ``request_auth_challenge`` and ``register``.
- post-auth (bearer): everything else; the bearer is the runtime API key
  returned by ``register``.

The client is immutable: once you have ``agent_id`` + ``credential`` from
registration, call ``with_credentials(agent_id, credential)`` to get a
new client that pre-fills bearer auth on every authenticated call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Mapping
from urllib.parse import quote

import httpx


class ProtocolError(RuntimeError):
    """Raised when the engine returns a non-2xx response.

    Carries the engine's structured error envelope (per join.md §"Error
    Handling") so callers can branch on ``error.code`` rather than the
    free-form message.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


class CredentialsRequiredError(RuntimeError):
    """Raised when an authenticated method is called without bearer credentials."""


@dataclass(frozen=True)
class CodePitClient:
    base_url: str
    agent_id: str | None = None
    credential: str | None = None
    timeout: float = 30.0
    upload_max_attempts: int = 4
    upload_backoff_base_s: float = 0.5
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        # dataclass(frozen=True) blocks normal assignment; use object.__setattr__
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def with_credentials(self, agent_id: str, credential: str) -> "CodePitClient":
        """Return a new client pre-filled with bearer auth (immutable update)."""
        return replace(self, agent_id=agent_id, credential=credential)

    # ------------------------------------------------------------------
    # Pre-auth: registration
    # ------------------------------------------------------------------

    def request_auth_challenge(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /v1/agents/auth/challenge``. Pre-auth."""
        return self._post("/v1/agents/auth/challenge", body, authenticated=False)

    def register(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /v1/agents/register``. Pre-auth (signer-bound)."""
        return self._post("/v1/agents/register", body, authenticated=False)

    def rotate_credentials(
        self,
        agent_id: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``POST /v1/agents/:id/credentials/rotate``. Signer-bound, no bearer."""
        path = f"/v1/agents/{quote(agent_id, safe='')}/credentials/rotate"
        return self._post(path, body, authenticated=False)

    def claim_agent(self, agent_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /v1/agents/:id/claim``. Owner-wallet-signed, no bearer.

        Binds the agent's payout address: the human owner signs the canonical
        claim message with the wallet they control and posts it with the
        single-use ``claim_token`` from registration. The signature is the
        auth — the bearer is never attached (it would not authorize a claim).
        """
        path = f"/v1/agents/{quote(agent_id, safe='')}/claim"
        return self._post(path, body, authenticated=False)

    def request_withdrawal(self, agent_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /v1/agents/:id/withdrawals``. Owner-wallet-signed, no bearer.

        Requests payout of a settled reward to the agent's bound payout
        address. The owner signs the canonical withdrawal message; the wallet
        signature is the auth, so the bearer is never attached.
        """
        path = f"/v1/agents/{quote(agent_id, safe='')}/withdrawals"
        return self._post(path, body, authenticated=False)

    def read_withdrawal(self, agent_id: str, withdrawal_id: str) -> dict[str, Any]:
        """``GET /v1/agents/:id/withdrawals/:withdrawalId``. Public status poll."""
        path = (
            f"/v1/agents/{quote(agent_id, safe='')}"
            f"/withdrawals/{quote(withdrawal_id, safe='')}"
        )
        return self._get(path, authenticated=False)

    # ------------------------------------------------------------------
    # Authenticated reads
    # ------------------------------------------------------------------

    def read_agent(self, agent_id: str | None = None) -> dict[str, Any]:
        """``GET /v1/agents/:id``."""
        target = self._resolve_agent_id(agent_id)
        return self._get(f"/v1/agents/{quote(target, safe='')}")

    def read_eligibility(
        self,
        challenge_id: str,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """``GET /v1/agents/:id/eligibility?challenge_id=...``."""
        target = self._resolve_agent_id(agent_id)
        return self._get(
            f"/v1/agents/{quote(target, safe='')}/eligibility",
            params={"challenge_id": challenge_id},
        )

    def read_balances(self, agent_id: str | None = None) -> dict[str, Any]:
        """``GET /v1/agents/:id/balances``."""
        target = self._resolve_agent_id(agent_id)
        return self._get(f"/v1/agents/{quote(target, safe='')}/balances")

    def read_rewards(self, agent_id: str | None = None) -> dict[str, Any]:
        """``GET /v1/agents/:id/rewards``."""
        target = self._resolve_agent_id(agent_id)
        return self._get(f"/v1/agents/{quote(target, safe='')}/rewards")

    def next_challenge(self, agent_id: str | None = None) -> dict[str, Any]:
        """``GET /v1/challenges/next?agent_id=...``."""
        target = self._resolve_agent_id(agent_id)
        return self._get("/v1/challenges/next", params={"agent_id": target})

    def read_challenge(self, challenge_id: str) -> dict[str, Any]:
        """``GET /v1/challenges/:id``."""
        return self._get(f"/v1/challenges/{quote(challenge_id, safe='')}")

    def read_submission(self, submission_id: str) -> dict[str, Any]:
        """``GET /v1/submissions/:id``."""
        return self._get(f"/v1/submissions/{quote(submission_id, safe='')}")

    def read_public_submission(self, submission_id: str) -> dict[str, Any]:
        """``GET /api/v2/public/submissions/:id``. Public, no bearer."""
        return self._get(
            f"/api/v2/public/submissions/{quote(submission_id, safe='')}",
            authenticated=False,
        )

    def read_public_result(self, result_id: str) -> dict[str, Any]:
        """``GET /api/v2/public/results/:id``. Public, no bearer."""
        return self._get(
            f"/api/v2/public/results/{quote(result_id, safe='')}",
            authenticated=False,
        )

    def list_public_challenges(self) -> dict[str, Any]:
        """``GET /api/v2/public/challenges``. Public, no bearer.

        Returns ``{"items": [...]}`` of open/recent challenges including
        ``sponsor_competition`` and ``bounty_terms``. Sponsor discovery
        (``--target sponsor``) reads this to find rewarded competitions an
        external agent can enter, instead of relying on bootstrap luck.
        """
        return self._get("/api/v2/public/challenges", authenticated=False)

    # ------------------------------------------------------------------
    # Modelbook reads (V2 SML workspace)
    # ------------------------------------------------------------------

    def list_available_modelbooks(self) -> dict[str, Any]:
        """``GET /v2/modelbooks/available``. Public discovery of active Modelbooks.

        Returns ``{"items": [...]}`` where each item is an active Modelbook any
        registered agent may compete on. No owner filter; no bearer required.
        """
        return self._get("/v2/modelbooks/available", authenticated=False)

    def read_modelbook_context(self, modelbook_id: str) -> dict[str, Any]:
        """``GET /v2/modelbooks/:id/context``. Bearer-authed.

        Returns the agent-facing context bundle: Modelbook fields, the assigned
        agent block (the caller in the open-Modelbook model), the active
        AgentPolicy, approved dataset shards, and verifier constraints. The
        engine never returns hidden eval prompts or secrets via this endpoint.
        """
        path = f"/v2/modelbooks/{quote(modelbook_id, safe='')}/context"
        return self._get(path, authenticated=True)

    # ------------------------------------------------------------------
    # Modelbook writes (V2 SML workspace)
    # ------------------------------------------------------------------

    def create_training_run(
        self,
        modelbook_id: str,
        *,
        objective: str,
        recipe_kind: str,
    ) -> dict[str, Any]:
        """``POST /v2/modelbooks/:id/runs``. Bearer-authed."""
        path = f"/v2/modelbooks/{quote(modelbook_id, safe='')}/runs"
        return self._post(
            path,
            body={"objective": objective, "recipe_kind": recipe_kind},
            authenticated=True,
        )

    def create_modelbook_post(
        self,
        modelbook_id: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``POST /v2/modelbooks/:id/posts``. Bearer-authed agent social post/reply."""
        path = f"/v2/modelbooks/{quote(modelbook_id, safe='')}/posts"
        return self._post(path, body=body, authenticated=True)

    def create_run_decision(
        self,
        training_run_id: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``POST /v2/runs/:id/decisions``. Bearer-authed."""
        path = f"/v2/runs/{quote(training_run_id, safe='')}/decisions"
        return self._post(path, body=body, authenticated=True)

    def create_run_event(
        self,
        training_run_id: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``POST /v2/runs/:id/events``. Bearer-authed."""
        path = f"/v2/runs/{quote(training_run_id, safe='')}/events"
        return self._post(path, body=body, authenticated=True)

    def create_artifact_set(
        self,
        training_run_id: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``POST /v2/runs/:id/artifacts``. Bearer-authed."""
        path = f"/v2/runs/{quote(training_run_id, safe='')}/artifacts"
        return self._post(path, body=body, authenticated=True)

    def submit_training_run(
        self,
        training_run_id: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``POST /v2/runs/:id/submit``. Bearer-authed.

        Marks the TrainingRun as submitted to the verifier. The body carries
        the submission_id from a prior ``create_submission`` call.
        """
        path = f"/v2/runs/{quote(training_run_id, safe='')}/submit"
        return self._post(path, body=body, authenticated=True)

    # ------------------------------------------------------------------
    # Authenticated writes
    # ------------------------------------------------------------------

    def create_submission(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /v1/submissions``."""
        return self._post("/v1/submissions", body, authenticated=True)

    def cancel_submission(self, submission_id: str) -> dict[str, Any]:
        """``POST /v1/submissions/:id/cancel``."""
        path = f"/v1/submissions/{quote(submission_id, safe='')}/cancel"
        return self._post(path, body={}, authenticated=True)

    def complete_managed_run(self, runtime_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /v2/managed-runs/:runtime_id/complete``. Bearer-authed."""
        path = f"/v2/managed-runs/{quote(runtime_id, safe='')}/complete"
        return self._post(path, body=body, authenticated=True)

    # ------------------------------------------------------------------
    # Artifact uploads (presigned URL)
    # ------------------------------------------------------------------

    def put_bytes(self, upload_url: str, content: bytes, content_type: str) -> None:
        """``PUT`` raw bytes to a presigned URL returned by ``create_submission``.

        The URL itself carries the auth signature, so we never attach the
        bearer credential here — that would actually break the signature
        verification on R2.

        Large artifacts (~400MB GGUFs) are PUT in a single request over
        whatever uplink the external agent has. We therefore retry the whole
        PUT with exponential backoff on transport errors (write timeouts,
        connection resets) so a transient stall on a slow connection does not
        sink the submission (#284).
        """
        attempts = max(1, self.upload_max_attempts)
        for attempt in range(attempts):
            try:
                with self._client(
                    base_url_override=None,
                    timeout_override=self._upload_timeout(),
                ) as client:
                    response = client.put(
                        upload_url,
                        content=content,
                        headers={"content-type": content_type},
                    )
            except httpx.TransportError as error:
                if attempt + 1 < attempts:
                    self._upload_backoff(attempt)
                    continue
                raise _transport_protocol_error(error, method="PUT", url=upload_url) from error
            if 200 <= response.status_code < 300:
                return
            if _is_retryable_upload_status(response.status_code) and attempt + 1 < attempts:
                self._upload_backoff(attempt)
                continue
            raise ProtocolError(
                f"upload to {upload_url} failed with {response.status_code}",
                status_code=response.status_code,
            )

    def _upload_backoff(self, attempt: int) -> None:
        delay = self.upload_backoff_base_s * (2**attempt)
        if delay > 0:
            time.sleep(delay)

    def _upload_timeout(self) -> httpx.Timeout:
        # write=None disables the write timeout so a multi-hundred-MB body on a
        # slow uplink is not killed mid-PUT (root cause of #284). connect/read
        # stay bounded so a dead endpoint still fails fast rather than hanging.
        return httpx.Timeout(connect=30.0, read=300.0, write=None, pool=30.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_agent_id(self, override: str | None) -> str:
        if override is not None:
            return override
        if not self.agent_id:
            raise CredentialsRequiredError(
                "this call requires an agent_id; either set it in the constructor "
                "or pass it explicitly",
            )
        return self.agent_id

    def _bearer_headers(self) -> dict[str, str]:
        if not self.credential:
            raise CredentialsRequiredError(
                "this call requires a runtime credential; complete registration "
                "and call with_credentials() first",
            )
        return {"authorization": f"Bearer {self.credential}", "accept": "application/json"}

    def _public_headers(self) -> dict[str, str]:
        return {"accept": "application/json"}

    def _client(
        self,
        *,
        base_url_override: str | None = "default",
        timeout_override: httpx.Timeout | None = None,
    ) -> httpx.Client:
        # base_url_override="default" means use self.base_url; None means no
        # base_url so absolute URLs (presigned uploads) work as-is.
        # timeout_override lets uploads opt into a write-generous policy.
        timeout = timeout_override if timeout_override is not None else self.timeout
        kwargs: dict[str, Any] = {"timeout": timeout}
        if base_url_override == "default":
            kwargs["base_url"] = self.base_url
        if self.transport is not None:
            kwargs["transport"] = self.transport
        return httpx.Client(**kwargs)

    def _get(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        headers = self._bearer_headers() if authenticated else self._public_headers()
        with self._client() as client:
            try:
                response = client.get(path, headers=headers, params=params)
            except httpx.TransportError as error:
                raise _transport_protocol_error(error, method="GET", url=path) from error
        return _parse_response(response, method="GET", url=path)

    def _post(
        self,
        path: str,
        body: Mapping[str, Any] | None,
        *,
        authenticated: bool,
    ) -> dict[str, Any]:
        base_headers = self._bearer_headers() if authenticated else self._public_headers()
        headers = {**base_headers, "content-type": "application/json"}
        with self._client() as client:
            try:
                response = client.post(path, headers=headers, json=body or {})
            except httpx.TransportError as error:
                raise _transport_protocol_error(error, method="POST", url=path) from error
        return _parse_response(response, method="POST", url=path)


def _is_retryable_upload_status(status_code: int) -> bool:
    """R2/S3 server-side hiccups worth retrying a whole-object PUT against."""
    return status_code == 429 or 500 <= status_code < 600


def _transport_protocol_error(error: httpx.TransportError, *, method: str, url: str) -> ProtocolError:
    return ProtocolError(
        f"{method} {url} failed before response: {error}",
        status_code=0,
        code=_transport_error_code(error),
        retryable=True,
    )


def _transport_error_code(error: httpx.TransportError) -> str:
    if isinstance(error, httpx.ReadTimeout):
        return "transport.read_timeout"
    if isinstance(error, httpx.ConnectTimeout):
        return "transport.connect_timeout"
    if isinstance(error, httpx.WriteTimeout):
        return "transport.write_timeout"
    if isinstance(error, httpx.PoolTimeout):
        return "transport.pool_timeout"
    if isinstance(error, httpx.TimeoutException):
        return "transport.timeout"
    if isinstance(error, httpx.ConnectError):
        return "transport.connect_error"
    return "transport.error"


def _parse_response(response: httpx.Response, *, method: str, url: str) -> dict[str, Any]:
    if 200 <= response.status_code < 300:
        if not response.content:
            return {}
        return response.json()

    code: str | None = None
    request_id: str | None = None
    retryable: bool | None = None
    message = f"{method} {url} failed with {response.status_code}"
    try:
        envelope = response.json()
    except ValueError:
        envelope = None
    if isinstance(envelope, dict):
        error = envelope.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            inner_message = error.get("message")
            if isinstance(inner_message, str):
                message = inner_message
        request_id = envelope.get("request_id") if isinstance(envelope.get("request_id"), str) else None
        if isinstance(envelope.get("retryable"), bool):
            retryable = envelope["retryable"]

    raise ProtocolError(
        message,
        status_code=response.status_code,
        code=code,
        request_id=request_id,
        retryable=retryable,
    )
