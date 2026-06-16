"""LLM client abstraction (spec §5.2 / §5.3).

The Data Factory never hardcodes a provider. Components receive an
:class:`LLMClient` and call :meth:`LLMClient.generate`. Two concrete clients are
provided:

* :class:`OpenAILLMClient` - thin adapter over the OpenAI-compatible / Ollama
  plumbing already used elsewhere in ``rxai_sdg`` (imported lazily so this module
  stays importable without the ``openai`` package installed).
* :class:`MockLLMClient` - a deterministic, scriptable client for unit and
  integration tests; supports response queues and callable handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable


@dataclass
class LLMResponse:
    """A single generation result.

    ``text`` is the message content. ``reasoning`` carries the model's chain of
    thought when the endpoint returns it in a **separate** field (e.g. the
    OpenAI-compatible ``message.reasoning_content`` emitted by reasoning models);
    it is ``None`` when the endpoint inlines reasoning in ``text`` (a ``<think>``
    block) or does not reason at all.

    ``logits`` optionally carries top-K logits per token for full-context
    distillation (spec §5.2, ``capture_logits``). It is ``None`` unless the
    client was asked to capture them and supports it.
    """

    text: str
    reasoning: Optional[str] = None
    logits: Optional[Any] = None
    raw: Optional[Any] = None


@runtime_checkable
class LLMClient(Protocol):
    """Protocol all LLM clients implement."""

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        capture_logits: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        ...


# ---------------------------------------------------------------------------
# OpenAI / Ollama adapter
# ---------------------------------------------------------------------------

class OpenAILLMClient:
    """Adapter that reuses :class:`rxai_sdg.base.BaseDatasetGenerator` plumbing.

    The heavy ``openai`` / ``ollama`` imports happen lazily inside ``__init__``
    so importing :mod:`rxai_sdg.factory` does not require those packages.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        use_ollama: bool = False,
        default_top_p: float = 0.9,
        log_first_raw: bool = False,
        reasoning_field_name: str = 'reasoning',
        timeout: float = 120,
    ):
        # Imported lazily to keep the factory package import-light.
        from ..base import BaseDatasetGenerator

        class _Backend(BaseDatasetGenerator):
            def _init_items(self) -> dict[str, list]:
                return {}

            def __call__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
                raise NotImplementedError

        self._backend = _Backend(
            model_name=model_name, api_url=api_url, api_key=api_key, use_ollama=use_ollama)
        self.model_name = model_name
        self.default_top_p = default_top_p
        #: log the first raw response so the reasoning-capture path can be verified
        #: against the live endpoint (field vs inline ``<think>``).
        self._log_first_raw = log_first_raw
        self._logged_raw = False
        self._reasoning_field_name = reasoning_field_name
        #: default per-request timeout (seconds); slow reasoning models under high
        #: concurrency need a generous value or calls time out and regenerate.
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        capture_logits: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        # Ollama path: the shared helper returns content only (reasoning, if any,
        # is inlined as a <think> block the Responder parses).
        if getattr(self._backend, "use_ollama", False):
            text = self._backend.generate_items(
                prompt=prompt, system_prompt=system_prompt, temperature=temperature,
                top_p=kwargs.get("top_p", self.default_top_p), max_tokens=max_tokens,
                stream=kwargs.get("stream", False),
                timeout=kwargs.get("timeout", self.timeout), additional_config={})
            return LLMResponse(text=text or "", reasoning=None)

        # OpenAI-compatible path: access the RAW response so we can capture
        # reasoning emitted in a separate ``message.reasoning_content`` field
        # (reasoning models such as Qwen3.5 do this); fall back to inline parsing
        # in the Responder when the field is absent.
        from ..base import default_additional_config

        try:
            completion = self._backend.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", self.default_top_p),
                timeout=kwargs.get("timeout", self.timeout),
            )
        except Exception as exc:  # pragma: no cover - network dependent
            print("API Error", exc)
            return LLMResponse(text="", reasoning=None)

        message = completion.choices[0].message
        text = getattr(message, "content", None) or ""
        reasoning = getattr(message, self._reasoning_field_name, None)
        if self._log_first_raw and not self._logged_raw:  # pragma: no cover
            self._logged_raw = True
            has_field = reasoning is not None
            print(f"[OpenAILLMClient] first raw response: {self._reasoning_field_name} "
                  f"present={has_field}; content_len={len(text)}; "
                  f"inline_think={'<think>' in text}")
        # Logit capture is not wired through the shared OpenAI helper; callers
        # who need it should subclass and use the provider's logprobs API.
        return LLMResponse(text=text, reasoning=reasoning, raw=completion)


# ---------------------------------------------------------------------------
# Deterministic mock client (testing)
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Scriptable, deterministic client for tests.

    Two scripting strategies, checked in order:

    1. ``handler`` - a callable ``(prompt, system_prompt, **kwargs) -> str``.
       This is the most flexible option and lets a test build responses that
       satisfy specific constraints (e.g. start every sentence with ``A``).
    2. ``responses`` - a queue of strings returned in order; once exhausted the
       last one repeats (or ``default`` is used if the queue was empty).
    """

    def __init__(
        self,
        responses: Optional[Sequence[Any]] = None,
        handler: Optional[Callable[..., Any]] = None,
        default: Any = "OK.",
        capture_logits_supported: bool = True,
    ):
        self._responses = list(responses or [])
        self._idx = 0
        self.handler = handler
        self.default = default
        self.capture_logits_supported = capture_logits_supported
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        capture_logits: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({
            "prompt": prompt, "system_prompt": system_prompt,
            "temperature": temperature, "max_tokens": max_tokens, "kwargs": kwargs,
        })
        if self.handler is not None:
            result = self.handler(prompt, system_prompt=system_prompt, **kwargs)
        elif self._responses:
            idx = min(self._idx, len(self._responses) - 1)
            result = self._responses[idx]
            self._idx += 1
        else:
            result = self.default
        # A scripted value may be a plain string (content only) or a full
        # ``LLMResponse`` - the latter lets tests set ``reasoning`` (the separate
        # ``reasoning_content`` field) to exercise the responder capture path.
        if isinstance(result, LLMResponse):
            resp = result
        else:
            resp = LLMResponse(text=str(result))
        if capture_logits and self.capture_logits_supported and resp.logits is None:
            resp.logits = {"mock": True, "tokens": len(resp.text.split())}
        return resp
