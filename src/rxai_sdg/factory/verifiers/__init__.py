"""Constraint verification subsystem (spec §5.4, §6.1, §9).

Importing this package registers all universal + English checkers in the global
registry. The public surface is :class:`ConstraintVerifier` plus the registry
helpers.
"""

from __future__ import annotations

from .base import (
    ConstraintVerifier,
    Checker,
    UnknownCheckerError,
    register_checker,
    register_language_stubs,
    resolve_checker,
    has_checker,
    registered_types,
    checker,
    WILDCARD_LANG,
)

# Importing these modules registers their checkers as a side effect.
from . import universal  # noqa: F401  (registration side effect)
from . import english  # noqa: F401  (registration side effect)

__all__ = [
    "ConstraintVerifier",
    "Checker",
    "UnknownCheckerError",
    "register_checker",
    "register_language_stubs",
    "resolve_checker",
    "has_checker",
    "registered_types",
    "checker",
    "WILDCARD_LANG",
]
