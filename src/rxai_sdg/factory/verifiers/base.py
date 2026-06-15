"""Constraint-verifier registry and the :class:`ConstraintVerifier` facade.

A *checker* is a callable ``(answer, params, conversation) -> (bool, detail)``.
Checkers are registered in a **language-aware registry** keyed by
``(constraint_type, lang)``. Universal checkers (``json_valid``,
``length_tokens``, ``n_bullets`` ...) register under the wildcard language
``"*"``; language-specific checkers (``first_letter``,
``no_gendered_pronouns`` ...) register under a concrete language such as
``"en"`` (spec §5.4 / §9).

Resolution order for ``(type, lang)``:

1. exact ``(type, lang)``
2. wildcard ``(type, "*")``

This lets a non-English locale override a single checker while inheriting the
universal ones.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..schemas import ConstraintSpec, VerifyResult

#: A checker returns ``(passed, human-readable detail)``.
Checker = Callable[[str, dict, Any], "tuple[bool, str]"]

WILDCARD_LANG = "*"

#: The global registry: ``{(constraint_type, lang): checker}``.
_REGISTRY: dict[tuple[str, str], Checker] = {}


class UnknownCheckerError(KeyError):
    """Raised when no checker is registered for a ``(type, lang)`` key."""


def register_checker(constraint_type: str, lang: str, checker: Checker) -> None:
    """Register ``checker`` for ``(constraint_type, lang)``."""
    _REGISTRY[(constraint_type, lang)] = checker


def checker(constraint_type: str, lang: str = WILDCARD_LANG) -> Callable[[Checker], Checker]:
    """Decorator form of :func:`register_checker`."""

    def deco(fn: Checker) -> Checker:
        register_checker(constraint_type, lang, fn)
        return fn

    return deco


def resolve_checker(constraint_type: str, lang: str) -> Checker:
    """Resolve a checker, falling back from ``lang`` to the wildcard language."""
    if (constraint_type, lang) in _REGISTRY:
        return _REGISTRY[(constraint_type, lang)]
    if (constraint_type, WILDCARD_LANG) in _REGISTRY:
        return _REGISTRY[(constraint_type, WILDCARD_LANG)]
    raise UnknownCheckerError(
        f"no checker registered for type={constraint_type!r} lang={lang!r}")


def has_checker(constraint_type: str, lang: str) -> bool:
    try:
        resolve_checker(constraint_type, lang)
        return True
    except UnknownCheckerError:
        return False


def registered_types(lang: Optional[str] = None) -> list[str]:
    """List registered constraint types, optionally filtered by language."""
    out = set()
    for (ctype, clang) in _REGISTRY:
        if lang is None or clang in (lang, WILDCARD_LANG):
            out.add(ctype)
    return sorted(out)


def _not_implemented_stub(constraint_type: str, lang: str) -> Checker:
    def stub(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
        raise NotImplementedError(
            f"checker {constraint_type!r} is not implemented for lang={lang!r}; "
            "implement and register a language-specific checker (spec §9)")

    return stub


def register_language_stubs(lang: str, constraint_types: list[str]) -> None:
    """Register explicit ``NotImplementedError`` stubs for a non-English locale.

    This makes the multilingual extension surface explicit: the registry knows a
    checker *should* exist for ``(type, lang)`` but raises a clear error until a
    native implementation is provided (spec §5.4 / §9).
    """
    for ctype in constraint_types:
        register_checker(ctype, lang, _not_implemented_stub(ctype, lang))


class ConstraintVerifier:
    """Per-response verifier (spec §5.4 / §6.1).

    Fires only when ``constraint_spec.verifier in {"programmatic", "hybrid"}``.
    For ``hybrid`` specs only the programmatic portion is checked here; the LLM
    fidelity portion (if any) is deferred to the optional holistic judge.
    """

    def verify(
        self,
        answer: str,
        constraint_spec: ConstraintSpec,
        conversation: Any = None,
    ) -> VerifyResult:
        if constraint_spec.verifier not in ("programmatic", "hybrid"):
            # Nothing machine-checkable here; treat as a pass at this stage.
            return VerifyResult(passed=True, detail="non-programmatic constraint; skipped")
        try:
            fn = resolve_checker(constraint_spec.type, constraint_spec.lang)
        except UnknownCheckerError as exc:
            return VerifyResult(passed=False, detail=f"unverifiable: {exc}")
        try:
            passed, detail = fn(answer, constraint_spec.params, conversation)
        except NotImplementedError as exc:
            return VerifyResult(passed=False, detail=f"not implemented: {exc}")
        except Exception as exc:  # defensive: a buggy checker must not crash the loop
            return VerifyResult(passed=False, detail=f"checker error: {exc!r}")
        return VerifyResult(passed=passed, detail=detail)
