"""Cache manager for KV cache entries with LRU eviction."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from prompt_cache_engine.exceptions import CacheFullError, EvictionError
from prompt_cache_engine.models import BatchAnalysis, CacheConfig, CacheStats, PrefixMatch
from prompt_cache_engine.trie import RadixTrie

logger = logging.getLogger(__name__)

# Estimated bytes per token for KV cache (K + V, fp16, typical hidden dim)
BYTES_PER_TOKEN_DEFAULT = 2048


@dataclass
class CacheEntry:
    """A single cached KV state entry.

    Args:
        cache_key: Unique identifier for this entry
        tokens: The token sequence this entry covers
        kv_data: The actual KV cache data (opaque to the cache manager)
        token_count: Number of tokens in this entry
        memory_bytes: Estimated memory usage in bytes
        created_at: Timestamp when entry was created
        last_accessed: Timestamp of last access
        access_count: Number of times this entry was accessed
    """

    cache_key: str
    tokens: Tuple[int, ...]
    kv_data: Any = None
    token_count: int = 0
    memory_bytes: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = len(self.tokens)
        if not self.memory_bytes:
            self.memory_bytes = self.token_count * BYTES_PER_TOKEN_DEFAULT
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_accessed:
            self.last_accessed = self.created_at


def _compute_cache_key(tokens: Tuple[int, ...]) -> str:
    """Compute a deterministic cache key from a token sequence.

    Args:
        tokens: Token sequence to hash

    Returns:
        Hex digest cache key
    """
    token_bytes = b"".join(t.to_bytes(4, byteorder="big", signed=True) for t in tokens)
    return hashlib.sha256(token_bytes).hexdigest()[:16]


class CacheManager:
    """Manages prefix-based KV cache entries with eviction.

    Combines a RadixTrie for efficient prefix matching with an LRU-ordered
    dict for eviction management. Supports configurable memory limits,
    TTL expiry, and batch prefix analysis.
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize the cache manager.

        Args:
            config: Cache configuration (uses defaults if not provided)
        """
        self.config = config or CacheConfig()
        self._trie = RadixTrie()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._total_memory_bytes = 0

        logger.info(
            f"CacheManager initialized: max_entries={self.config.max_entries}, "
            f"max_memory_mb={self.config.max_memory_mb}, "
            f"policy={self.config.eviction_policy}"
        )

    @property
    def stats(self) -> CacheStats:
        """Get current cache statistics."""
        self._stats.entries_count = len(self._entries)
        self._stats.memory_used_mb = self._total_memory_bytes / (1024 * 1024)
        return self._stats

    def lookup(self, tokens: Tuple[int, ...]) -> PrefixMatch:
        """Look up the longest cached prefix for a token sequence.

        Args:
            tokens: Token sequence to look up

        Returns:
            PrefixMatch describing the match result
        """
        self._stats.total_lookups += 1
        self._stats.total_tokens_requested += len(tokens)

        matched_len, cache_key = self._trie.find_longest_prefix(tokens)

        if cache_key is None or matched_len < self.config.min_prefix_length:
            self._stats.cache_misses += 1
            return PrefixMatch(
                total_length=len(tokens),
                hit=False,
            )

        # Check entry exists and is not expired
        entry = self._entries.get(cache_key)
        if entry is None:
            self._stats.cache_misses += 1
            return PrefixMatch(total_length=len(tokens), hit=False)

        if self._is_expired(entry):
            self._evict_entry(cache_key)
            self._stats.cache_misses += 1
            return PrefixMatch(total_length=len(tokens), hit=False)

        # Update access tracking
        entry.last_accessed = time.time()
        entry.access_count += 1
        self._entries.move_to_end(cache_key)

        self._stats.cache_hits += 1
        self._stats.total_tokens_served += matched_len

        return PrefixMatch(
            matched_tokens=tokens[:matched_len],
            matched_length=matched_len,
            total_length=len(tokens),
            cache_key=cache_key,
            remaining_tokens=tokens[matched_len:],
            hit=True,
        )

    def store(
        self,
        tokens: Tuple[int, ...],
        kv_data: Any = None,
        memory_bytes: int = 0,
    ) -> str:
        """Store a KV cache entry for a token sequence.

        Args:
            tokens: Token sequence this KV data covers
            kv_data: The actual KV cache data (opaque)
            memory_bytes: Override memory estimate (0 = auto-estimate)

        Returns:
            Cache key for the stored entry

        Raises:
            CacheFullError: If entry cannot be stored after eviction
        """
        if len(tokens) < self.config.min_prefix_length:
            logger.debug(f"Skipping store: {len(tokens)} tokens < min {self.config.min_prefix_length}")
            return ""

        cache_key = _compute_cache_key(tokens)

        # Check if already cached
        if cache_key in self._entries:
            entry = self._entries[cache_key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._entries.move_to_end(cache_key)
            return cache_key

        entry = CacheEntry(
            cache_key=cache_key,
            tokens=tokens,
            kv_data=kv_data,
            memory_bytes=memory_bytes or len(tokens) * BYTES_PER_TOKEN_DEFAULT,
        )

        # Evict if needed
        self._ensure_capacity(entry.memory_bytes)

        # Store entry
        self._entries[cache_key] = entry
        self._trie.insert(tokens, cache_key)
        self._total_memory_bytes += entry.memory_bytes

        logger.debug(
            f"Stored entry: key={cache_key}, tokens={len(tokens)}, "
            f"memory={entry.memory_bytes / 1024:.1f}KB"
        )
        return cache_key

    def evict(self, cache_key: str) -> bool:
        """Manually evict a specific cache entry.

        Args:
            cache_key: Key of the entry to evict

        Returns:
            True if entry was evicted, False if not found
        """
        return self._evict_entry(cache_key)

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._entries)
        self._entries.clear()
        self._trie = RadixTrie()
        self._total_memory_bytes = 0
        logger.info(f"Cache cleared: {count} entries removed")

    def analyze_batch(
        self,
        token_sequences: List[Tuple[int, ...]],
    ) -> BatchAnalysis:
        """Analyze prefix sharing potential within a batch.

        Identifies common prefixes across a batch of token sequences
        and calculates potential deduplication savings.

        Args:
            token_sequences: List of token sequences to analyze

        Returns:
            BatchAnalysis with sharing statistics
        """
        if not token_sequences:
            return BatchAnalysis()

        # Build a temporary trie for analysis
        analysis_trie: Dict[Tuple[int, ...], List[int]] = {}

        for idx, tokens in enumerate(token_sequences):
            # Find the longest prefix shared with any other sequence
            for length in range(len(tokens), self.config.min_prefix_length - 1, -1):
                prefix = tokens[:length]
                if prefix not in analysis_trie:
                    analysis_trie[prefix] = []
                analysis_trie[prefix].append(idx)

        # Find maximal shared prefixes (prefixes appearing 2+ times)
        shared_groups: Dict[str, List[int]] = {}
        assigned: Dict[int, int] = {}  # idx -> best prefix length

        sorted_prefixes = sorted(analysis_trie.keys(), key=len, reverse=True)
        for prefix in sorted_prefixes:
            indices = analysis_trie[prefix]
            if len(indices) < 2:
                continue
            # Only consider indices not yet assigned to a longer prefix
            unassigned = [i for i in indices if i not in assigned]
            if len(unassigned) < 2:
                continue
            prefix_key = _compute_cache_key(prefix)[:8]
            shared_groups[prefix_key] = unassigned
            for i in unassigned:
                assigned[i] = len(prefix)

        total_tokens = sum(len(t) for t in token_sequences)
        savings = sum(
            assigned.get(i, 0) for i in range(len(token_sequences))
        )
        # Subtract one copy per group (must compute at least once)
        for group_indices in shared_groups.values():
            if group_indices:
                max_savings_in_group = max(
                    assigned.get(i, 0) for i in group_indices
                )
                savings -= max_savings_in_group

        return BatchAnalysis(
            batch_size=len(token_sequences),
            unique_prefixes=len(shared_groups),
            shared_prefix_groups=shared_groups,
            potential_savings_tokens=max(0, savings),
            total_tokens=total_tokens,
        )

    def get_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key without updating access tracking.

        Args:
            cache_key: Key of the entry

        Returns:
            CacheEntry if found, None otherwise
        """
        return self._entries.get(cache_key)

    def _ensure_capacity(self, needed_bytes: int) -> None:
        """Ensure there is capacity for a new entry.

        Args:
            needed_bytes: Memory bytes needed for the new entry

        Raises:
            CacheFullError: If eviction cannot free enough space
        """
        max_memory_bytes = int(self.config.max_memory_mb * 1024 * 1024)
        eviction_attempts = 0
        max_attempts = len(self._entries) + 1

        while (
            len(self._entries) >= self.config.max_entries
            or self._total_memory_bytes + needed_bytes > max_memory_bytes
        ) and self._entries:
            eviction_attempts += 1
            if eviction_attempts > max_attempts:
                raise CacheFullError(
                    f"Cannot free enough space after {eviction_attempts} evictions"
                )
            self._evict_one()

    def _evict_one(self) -> None:
        """Evict the least recently used entry."""
        if not self._entries:
            return

        if self.config.eviction_policy == "lru":
            # OrderedDict: first item is LRU
            oldest_key = next(iter(self._entries))
            self._evict_entry(oldest_key)
        elif self.config.eviction_policy == "lfu":
            # Find least frequently used
            lfu_key = min(self._entries, key=lambda k: self._entries[k].access_count)
            self._evict_entry(lfu_key)

    def _evict_entry(self, cache_key: str) -> bool:
        """Evict a specific entry.

        Args:
            cache_key: Key of the entry to evict

        Returns:
            True if evicted, False if not found
        """
        entry = self._entries.pop(cache_key, None)
        if entry is None:
            return False

        self._trie.remove(entry.tokens)
        self._total_memory_bytes -= entry.memory_bytes
        self._stats.evictions += 1

        logger.debug(f"Evicted entry: key={cache_key}, tokens={entry.token_count}")
        return True

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired.

        Args:
            entry: Entry to check

        Returns:
            True if expired
        """
        if self.config.default_ttl_seconds <= 0:
            return False
        return (time.time() - entry.created_at) > self.config.default_ttl_seconds
