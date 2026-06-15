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

    ``logits`` optionally carries top-K logits per token for full-context
    distillation (spec §5.2, ``capture_logits``). It is ``None`` unless the
    client was asked to capture them and supports it.
    """

    text: str
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
        text = self._backend.generate_items(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=kwargs.get("top_p", self.default_top_p),
            max_tokens=max_tokens,
            stream=kwargs.get("stream", False),
            timeout=kwargs.get("timeout", 120),
        )
        # Logit capture is not wired through the shared OpenAI helper; callers
        # who need it should subclass and use the provider's logprobs API.
        return LLMResponse(text=text, logits=None)


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
        responses: Optional[Sequence[str]] = None,
        handler: Optional[Callable[..., str]] = None,
        default: str = "OK.",
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
            "temperature": temperature, "kwargs": kwargs,
        })
        if self.handler is not None:
            text = self.handler(prompt, system_prompt=system_prompt, **kwargs)
        elif self._responses:
            idx = min(self._idx, len(self._responses) - 1)
            text = self._responses[idx]
            self._idx += 1
        else:
            text = self.default
        logits = None
        if capture_logits and self.capture_logits_supported:
            logits = {"mock": True, "tokens": len(text.split())}
        return LLMResponse(text=text, logits=logits)
