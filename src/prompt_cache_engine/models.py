"""Data models for prompt-cache-engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CacheConfig:
    """Configuration for the cache manager.

    Args:
        max_entries: Maximum number of cached KV entries
        max_memory_mb: Maximum memory budget in megabytes
        default_ttl_seconds: Default time-to-live for cache entries (0 = no expiry)
        eviction_policy: Cache eviction policy ("lru" or "lfu")
        min_prefix_length: Minimum token length to consider for caching
    """

    max_entries: int = 10000
    max_memory_mb: float = 1024.0
    default_ttl_seconds: float = 0.0
    eviction_policy: str = "lru"
    min_prefix_length: int = 4

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be > 0, got {self.max_memory_mb}")
        if self.eviction_policy not in ("lru", "lfu"):
            raise ValueError(
                f"eviction_policy must be 'lru' or 'lfu', got '{self.eviction_policy}'"
            )
        if self.min_prefix_length < 1:
            raise ValueError(
                f"min_prefix_length must be >= 1, got {self.min_prefix_length}"
            )


@dataclass
class PrefixMatch:
    """Result of a prefix lookup in the cache.

    Args:
        matched_tokens: The token sequence that matched a cached prefix
        matched_length: Number of tokens in the matched prefix
        total_length: Total number of tokens in the query
        cache_key: Key identifying the cached KV entry
        remaining_tokens: Tokens not covered by the cached prefix
        hit: Whether a cache hit occurred
    """

    matched_tokens: Tuple[int, ...] = ()
    matched_length: int = 0
    total_length: int = 0
    cache_key: str = ""
    remaining_tokens: Tuple[int, ...] = ()
    hit: bool = False

    @property
    def savings_ratio(self) -> float:
        """Fraction of tokens saved by the cache hit."""
        if self.total_length == 0:
            return 0.0
        return self.matched_length / self.total_length


@dataclass
class CacheStats:
    """Aggregated cache statistics.

    Args:
        total_lookups: Total number of cache lookups
        cache_hits: Number of lookups that found a prefix match
        cache_misses: Number of lookups with no match
        total_tokens_served: Total tokens served from cache
        total_tokens_requested: Total tokens across all lookups
        entries_count: Current number of cache entries
        memory_used_mb: Current estimated memory usage
        evictions: Total number of cache evictions
    """

    total_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_served: int = 0
    total_tokens_requested: int = 0
    entries_count: int = 0
    memory_used_mb: float = 0.0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        if self.total_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_lookups

    @property
    def token_savings_rate(self) -> float:
        """Fraction of total tokens served from cache."""
        if self.total_tokens_requested == 0:
            return 0.0
        return self.total_tokens_served / self.total_tokens_requested


@dataclass
class BatchAnalysis:
    """Analysis of prefix sharing within a batch of prompts.

    Args:
        batch_size: Number of prompts in the batch
        unique_prefixes: Number of distinct prefixes found
        shared_prefix_groups: Groups of prompts sharing the same prefix
        potential_savings_tokens: Tokens that could be deduplicated
        total_tokens: Total tokens across all prompts
    """

    batch_size: int = 0
    unique_prefixes: int = 0
    shared_prefix_groups: Dict[str, List[int]] = field(default_factory=dict)
    potential_savings_tokens: int = 0
    total_tokens: int = 0

    @property
    def dedup_ratio(self) -> float:
        """Fraction of tokens that can be deduplicated."""
        if self.total_tokens == 0:
            return 0.0
        return self.potential_savings_tokens / self.total_tokens
