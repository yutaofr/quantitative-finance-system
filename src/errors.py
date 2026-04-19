"""Shared exceptions for SRD v8.7 contracts."""

from __future__ import annotations


class LawEngineError(Exception):
    """Base class for law engine errors."""


class VintageUnavailableError(LawEngineError):
    """Raised when strict PIT data is requested before its first vintage."""


class HMMConvergenceError(LawEngineError):
    """Raised when HMM EM cannot return a converged production model."""
