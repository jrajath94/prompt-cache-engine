"""Custom exceptions for prompt-cache-engine."""

from __future__ import annotations


class CacheError(Exception):
    """Base exception for all cache errors."""


class CacheFullError(CacheError):
    """Raised when cache is at capacity and eviction fails."""


class EvictionError(CacheError):
    """Raised when cache eviction encounters an error."""


class TokenizationError(CacheError):
    """Raised when token conversion fails."""
