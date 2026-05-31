"""Optimizer Brain — LLM-backed optimization planning and result interpretation.

The Brain is the optimizer's strategic seam: instead of always running one
hard-coded recipe pipeline left-to-right, an agent owner can plug in an
LLM-backed ``Brain`` that proposes bounded optimization experiments based on
the challenge spec and the agent's recent history.

V1 active provider: managed (engine-routed). BYOK providers ship as stubs
that raise ``NotImplementedError`` — we keep the import surface stable so
Phase B can flip them on with a config change, not a refactor.

The Brain accepts dependency-injected providers so tests can drop in a
canned-response stub without touching real network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .plan import (
    OPTIMIZATION_PLAN_SCHEMA,
    OptimizationPlan,
    OptimizationPlanError,
    parse_optimization_plan,
    safe_default_plan,
)
from .prompts import (
    KNOWN_RECIPE_NAMES,
    RECIPE_CHOICE_SCHEMA,
    build_optimization_plan_prompt,
    build_pick_recipe_prompt,
)

# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class BrainProvider(Protocol):
    """Minimal seam every brain provider implements.

    A provider takes a prompt + an optional JSON schema and returns the
    raw model output. The ``Brain`` is responsible for parsing JSON and
    validating against the schema — providers MUST NOT swallow malformed
    output silently.
    """

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
        """Return raw model output (text or JSON string)."""
        ...


class BrainError(RuntimeError):
    """Raised by the Brain on schema-violation, provider-failure, or
    unrecoverable LLM output."""


# ---------------------------------------------------------------------------
# Recipe choice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeChoice:
    recipe_name: str
    confidence: float = 0.0
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.recipe_name not in KNOWN_RECIPE_NAMES:
            raise BrainError(
                f"recipe_name {self.recipe_name!r} is not a known recipe; "
                f"expected one of {KNOWN_RECIPE_NAMES}",
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise BrainError(
                f"confidence must be in [0,1] (got {self.confidence})",
            )


@dataclass(frozen=True)
class TinyChatQuantizationChoice:
    quantization_profile: str
    confidence: float = 0.0
    reasoning: str = ""

    def __post_init__(self) -> None:
        if not self.quantization_profile.strip():
            raise BrainError("quantization_profile must not be blank")
        if not (0.0 <= self.confidence <= 1.0):
            raise BrainError(
                f"confidence must be in [0,1] (got {self.confidence})",
            )


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrainConfig:
    """Per-agent Brain configuration.

    ``tier`` is one of "cheap" | "mid" | "premium" | "network" — matches the
    engine's brain tier table. ``provider_name`` is informational only;
    routing is determined by the ``BrainProvider`` instance the Brain holds.

    ``max_retries_per_step`` caps how many times ``should_retry`` is allowed
    to recommend retrying a single failing step before the orchestrator
    bails out. ``fallback_on_error`` keeps the older safe-default behavior;
    external autonomous agents can turn it off so a missing LLM brain fails
    loudly instead of pretending a deterministic recipe was strategic.
    """

    provider_name: str = "managed"
    tier: str = "cheap"
    max_retries_per_step: int = 2
    fallback_on_error: bool = True
    action_id_prefix: str | None = None

    def __post_init__(self) -> None:
        if self.tier not in {"cheap", "mid", "premium", "network"}:
            raise BrainError(
                f"tier must be one of cheap|mid|premium|network (got {self.tier!r})",
            )
        if self.max_retries_per_step < 0:
            raise BrainError(
                "max_retries_per_step must be >= 0",
            )
        if self.action_id_prefix is not None and not self.action_id_prefix.strip():
            raise BrainError("action_id_prefix must not be blank")


@dataclass
class Brain:
    """LLM-backed strategic seam for the optimizer.

    Construct via :meth:`with_stub_responses` for tests, or with an actual
    ``BrainProvider`` instance for production.
    """

    config: BrainConfig
    provider: BrainProvider
    _action_counter: int = field(default=0, init=False)

    # ----- recipe picking --------------------------------------------------

    def plan_optimization(
        self,
        challenge_spec: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] = (),
    ) -> OptimizationPlan:
        """Ask the LLM for a bounded optimization experiment plan.

        This is the preferred LLM control surface. It gives the model room to
        reason about multiple candidate experiments while keeping execution
        inside validated transform primitives. Invalid provider output falls
        back to baseline export unless strict mode is enabled.
        """
        prompt = build_optimization_plan_prompt(challenge_spec, history)
        action_id = self._next_action_id("plan_optimization")
        try:
            raw = self.provider.generate(
                prompt=prompt,
                action_id=action_id,
                attempt=1,
                tier=self.config.tier,
                schema=OPTIMIZATION_PLAN_SCHEMA,
            )
        except Exception as error:
            if not self.config.fallback_on_error:
                raise BrainError(f"brain provider error: {error}") from error
            return safe_default_plan(reason=f"brain provider error: {error}")

        try:
            return parse_optimization_plan(raw)
        except OptimizationPlanError as error:
            if not self.config.fallback_on_error:
                raise BrainError(f"brain plan rejected: {error}") from error
            return safe_default_plan(reason=f"brain plan rejected: {error}")

    def pick_recipe(
        self,
        challenge_spec: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] = (),
    ) -> RecipeChoice:
        """Ask the LLM which recipe to run for this challenge.

        Falls back to ``baseline-export`` on parse / validation failure when
        ``fallback_on_error`` is enabled. In strict mode, provider or parsing
        failures raise ``BrainError`` so the orchestrator cannot silently run
        a non-LLM path.
        """
        prompt = build_pick_recipe_prompt(challenge_spec, history)
        action_id = self._next_action_id("pick_recipe")
        try:
            raw = self.provider.generate(
                prompt=prompt,
                action_id=action_id,
                attempt=1,
                tier=self.config.tier,
                schema=RECIPE_CHOICE_SCHEMA,
            )
        except Exception as error:
            if not self.config.fallback_on_error:
                raise BrainError(f"brain provider error: {error}") from error
            return RecipeChoice(
                recipe_name="baseline-export",
                confidence=0.0,
                reasoning=f"brain provider error: {error}",
            )

        try:
            parsed = _parse_recipe_choice(raw)
        except BrainError as error:
            if not self.config.fallback_on_error:
                raise
            return RecipeChoice(
                recipe_name="baseline-export",
                confidence=0.0,
                reasoning=f"brain output rejected: {error}",
            )
        return parsed

    def pick_tiny_chat_quantization(
        self,
        *,
        challenge_spec: Mapping[str, Any],
        allowed_profiles: Sequence[str],
        current_profile: str,
        history: Sequence[Mapping[str, Any]] = (),
    ) -> TinyChatQuantizationChoice:
        """Ask the LLM for a bounded GGUF quantization choice for Qwen tiny-chat.

        This is intentionally narrower than ``plan_optimization``. The current
        launch lane can safely vary GGUF quantization, but it cannot execute
        arbitrary training code or unbounded model edits.
        """

        profiles = tuple(profile for profile in allowed_profiles if profile.strip())
        if current_profile not in profiles:
            raise BrainError(
                f"current_profile {current_profile!r} is not in allowed_profiles",
            )
        schema = _tiny_chat_quantization_schema(profiles)
        prompt = _build_tiny_chat_quantization_prompt(
            challenge_spec=challenge_spec,
            allowed_profiles=profiles,
            current_profile=current_profile,
            history=history,
        )
        action_id = self._next_action_id("tiny_chat_quantization")
        try:
            raw = self.provider.generate(
                prompt=prompt,
                action_id=action_id,
                attempt=1,
                tier=self.config.tier,
                schema=schema,
            )
        except Exception as error:
            if not self.config.fallback_on_error:
                raise BrainError(f"brain provider error: {error}") from error
            return TinyChatQuantizationChoice(
                quantization_profile=current_profile,
                confidence=0.0,
                reasoning=f"brain provider error: {error}",
            )

        try:
            return _parse_tiny_chat_quantization_choice(raw, allowed_profiles=profiles)
        except BrainError as error:
            if not self.config.fallback_on_error:
                raise
            return TinyChatQuantizationChoice(
                quantization_profile=current_profile,
                confidence=0.0,
                reasoning=f"brain output rejected: {error}",
            )

    # ----- result interpretation ------------------------------------------

    def interpret_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize a verifier outcome as a dict the orchestrator stores
        in its history blob.

        Pure-function for now — no LLM call. Phase B will optionally route
        ambiguous outcomes through the Brain for narrative summarization.
        """
        state = str(result.get("state") or "UNKNOWN").upper()
        outcome: str
        if state in {"VERIFIED", "SETTLED", "PUBLISHED"}:
            outcome = "success"
        elif state in {"VALIDATION_FAILED", "BENCHMARK_FAILED"}:
            outcome = "failure"
        elif state in {"INVALIDATED", "CANCELLED"}:
            outcome = "cancelled"
        else:
            outcome = "unknown"
        return {
            "state": state,
            "outcome": outcome,
            "submission_id": result.get("submission_id"),
            "proof_record_id": result.get("proof_record_id"),
        }

    # ----- retry policy ----------------------------------------------------

    def should_retry(
        self,
        *,
        attempt: int,
        error_code: str | None,
        retryable: bool | None,
    ) -> bool:
        """Decide whether the orchestrator should retry a failing step.

        Conservative policy: never retry past ``max_retries_per_step``;
        always retry if the protocol error explicitly says ``retryable``;
        never retry on terminal codes (``brain.invalid_input``, hard auth
        failures); retry on transient codes.
        """
        if attempt > self.config.max_retries_per_step:
            return False
        if retryable is True:
            return True
        if retryable is False:
            return False
        # Unknown retryable hint — fall back to error-code heuristics.
        if error_code is None:
            return False
        terminal_prefixes = (
            "auth.",
            "agent.suspended",
            "agent.ineligible",
            "submission.invalid_state",
            "submission.idempotency_conflict",
            "brain.invalid_input",
            "brain.insufficient_balance",
        )
        if any(error_code.startswith(p) for p in terminal_prefixes):
            return False
        return True

    # ----- test seam -------------------------------------------------------

    @classmethod
    def with_stub_responses(
        cls,
        responses: Sequence[str | Mapping[str, Any]],
        *,
        config: BrainConfig | None = None,
    ) -> "Brain":
        """Build a Brain whose provider returns canned responses in order.

        Each response may be a raw string or a Mapping (which is JSON-encoded
        before handing back to the Brain). This lets tests assert that the
        Brain handles both raw-text and structured-JSON responses through
        the same code path.
        """
        cfg = config or BrainConfig()
        return cls(config=cfg, provider=_StubProvider(list(responses)))

    # ----- internals -------------------------------------------------------

    def _next_action_id(self, kind: str) -> str:
        self._action_counter += 1
        action_id = f"{kind}-{self._action_counter:04d}"
        if self.config.action_id_prefix:
            return f"{self.config.action_id_prefix}:{action_id}"
        return action_id


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_recipe_choice(raw: str) -> RecipeChoice:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BrainError(f"brain returned non-JSON output: {error}") from error
    if not isinstance(payload, Mapping):
        raise BrainError(
            f"brain output was not a JSON object: {type(payload).__name__}",
        )
    recipe_name = payload.get("recipe_name")
    if not isinstance(recipe_name, str):
        raise BrainError("brain output missing 'recipe_name' string")
    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as error:
        raise BrainError(
            f"brain output 'confidence' is not a number: {confidence_raw!r}",
        ) from error
    reasoning = str(payload.get("reasoning") or "")
    return RecipeChoice(
        recipe_name=recipe_name,
        confidence=confidence,
        reasoning=reasoning,
    )


def _tiny_chat_quantization_schema(profiles: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["quantization_profile", "rationale"],
        "properties": {
            "quantization_profile": {"type": "string", "enum": list(profiles)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 1024},
        },
        "additionalProperties": False,
    }


def _build_tiny_chat_quantization_prompt(
    *,
    challenge_spec: Mapping[str, Any],
    allowed_profiles: Sequence[str],
    current_profile: str,
    history: Sequence[Mapping[str, Any]],
) -> str:
    challenge_blob = json.dumps(challenge_spec, sort_keys=True, indent=2, default=str)
    history_blob = json.dumps(
        [dict(item) for item in history],
        sort_keys=True,
        indent=2,
        default=str,
    )
    return (
        "You are CodePit's launch managed-agent brain for a Qwen tiny-chat "
        "GGUF competition. Choose exactly one supported quantization profile "
        "for the next candidate artifact. Prefer preserving benchmark quality "
        "while still improving deployability. Do not invent profiles.\n\n"
        f"Allowed quantization profiles: {', '.join(allowed_profiles)}\n"
        f"Current/default profile: {current_profile}\n\n"
        "Challenge spec:\n"
        f"{challenge_blob}\n\n"
        "Recent compact history:\n"
        f"{history_blob}\n\n"
        "Return JSON only with quantization_profile, confidence, and rationale."
    )


def _parse_tiny_chat_quantization_choice(
    raw: str,
    *,
    allowed_profiles: Sequence[str],
) -> TinyChatQuantizationChoice:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BrainError(f"brain returned non-JSON output: {error}") from error
    if not isinstance(payload, Mapping):
        raise BrainError(
            f"brain output was not a JSON object: {type(payload).__name__}",
        )
    profile = payload.get("quantization_profile")
    if not isinstance(profile, str):
        raise BrainError("brain output missing 'quantization_profile' string")
    if profile not in allowed_profiles:
        raise BrainError(
            f"brain chose unsupported quantization_profile {profile!r}; "
            f"expected one of {tuple(allowed_profiles)!r}",
        )
    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as error:
        raise BrainError(
            f"brain output 'confidence' is not a number: {confidence_raw!r}",
        ) from error
    return TinyChatQuantizationChoice(
        quantization_profile=profile,
        confidence=confidence,
        reasoning=str(payload.get("rationale") or payload.get("reasoning") or ""),
    )


class _StubProvider:
    """Test-only ``BrainProvider`` that replays a fixed sequence."""

    def __init__(self, responses: list[str | Mapping[str, Any]]) -> None:
        self._responses = list(responses)

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
        if not self._responses:
            raise BrainError("stub Brain ran out of canned responses")
        next_response = self._responses.pop(0)
        if isinstance(next_response, Mapping):
            return json.dumps(dict(next_response))
        return next_response


__all__ = [
    "Brain",
    "BrainConfig",
    "BrainError",
    "BrainProvider",
    "OptimizationPlan",
    "RecipeChoice",
    "TinyChatQuantizationChoice",
]
