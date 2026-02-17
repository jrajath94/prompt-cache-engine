"""prompt-cache-engine: Engine-agnostic KV cache sharing for prompt prefix deduplication."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Rajath John"

from prompt_cache_engine.cache import CacheEntry, CacheManager
from prompt_cache_engine.exceptions import (
    CacheError,
    CacheFullError,
    EvictionError,
    TokenizationError,
)
from prompt_cache_engine.models import CacheConfig, CacheStats, PrefixMatch
from prompt_cache_engine.trie import RadixTrie, TrieNode

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "CacheError",
    "CacheFullError",
    "CacheManager",
    "CacheStats",
    "EvictionError",
    "PrefixMatch",
    "RadixTrie",
    "TokenizationError",
    "TrieNode",
]
