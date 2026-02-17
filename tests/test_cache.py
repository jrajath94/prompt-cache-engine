"""Tests for cache manager."""

from __future__ import annotations

import time

import pytest

from prompt_cache_engine.cache import CacheManager, _compute_cache_key
from prompt_cache_engine.exceptions import CacheFullError
from prompt_cache_engine.models import CacheConfig


class TestCacheKey:
    """Tests for cache key computation."""

    def test_deterministic(self) -> None:
        """Same tokens produce same key."""
        tokens = (1, 2, 3, 4, 5)
        assert _compute_cache_key(tokens) == _compute_cache_key(tokens)

    def test_different_tokens_different_keys(self) -> None:
        """Different tokens produce different keys."""
        key_a = _compute_cache_key((1, 2, 3))
        key_b = _compute_cache_key((4, 5, 6))
        assert key_a != key_b

    def test_key_length(self) -> None:
        """Cache key is 16 characters."""
        key = _compute_cache_key((1, 2, 3))
        assert len(key) == 16


class TestCacheManagerLookupAndStore:
    """Tests for lookup and store operations."""

    def test_miss_on_empty_cache(self, small_cache: CacheManager) -> None:
        """Empty cache returns a miss."""
        match = small_cache.lookup((1, 2, 3, 4, 5))
        assert not match.hit
        assert match.matched_length == 0

    def test_store_and_lookup(self, small_cache: CacheManager) -> None:
        """Store tokens then lookup returns hit."""
        tokens = (1, 2, 3, 4, 5)
        key = small_cache.store(tokens)
        assert key != ""

        match = small_cache.lookup(tokens)
        assert match.hit
        assert match.matched_length == 5
        assert match.cache_key == key

    def test_prefix_hit(self, small_cache: CacheManager) -> None:
        """Stored prefix matches longer query."""
        small_cache.store((1, 2, 3, 4))
        match = small_cache.lookup((1, 2, 3, 4, 5, 6, 7, 8))
        assert match.hit
        assert match.matched_length == 4
        assert match.remaining_tokens == (5, 6, 7, 8)

    def test_min_prefix_length_enforced_on_store(self, small_cache: CacheManager) -> None:
        """Tokens shorter than min_prefix_length are not stored."""
        key = small_cache.store((1,))  # min_prefix_length is 2
        assert key == ""

    def test_min_prefix_length_enforced_on_lookup(self, small_cache: CacheManager) -> None:
        """Match shorter than min_prefix_length returns miss."""
        small_cache.store((1, 2, 3, 4))
        # With min_prefix_length=2, a match of length 1 should be treated as miss
        match = small_cache.lookup((1, 99))
        assert not match.hit

    def test_duplicate_store_updates_access(self, small_cache: CacheManager) -> None:
        """Storing the same tokens again updates access time."""
        tokens = (1, 2, 3, 4, 5)
        key1 = small_cache.store(tokens)
        key2 = small_cache.store(tokens)
        assert key1 == key2

        entry = small_cache.get_entry(key1)
        assert entry is not None
        assert entry.access_count == 1  # incremented on second store

    def test_stats_tracking(self, small_cache: CacheManager) -> None:
        """Stats are correctly updated."""
        tokens = (1, 2, 3, 4, 5)
        small_cache.store(tokens)

        small_cache.lookup(tokens)  # hit
        small_cache.lookup((10, 20, 30, 40, 50))  # miss

        stats = small_cache.stats
        assert stats.total_lookups == 2
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.total_tokens_served == 5
        assert stats.total_tokens_requested == 10


class TestCacheManagerEviction:
    """Tests for eviction behavior."""

    def test_lru_eviction(self) -> None:
        """LRU eviction removes oldest entry."""
        config = CacheConfig(max_entries=2, min_prefix_length=2)
        cache = CacheManager(config=config)

        cache.store((1, 2, 3))
        cache.store((4, 5, 6))
        # Cache is full. Storing a third should evict (1,2,3)
        cache.store((7, 8, 9))

        match_old = cache.lookup((1, 2, 3))
        match_new = cache.lookup((7, 8, 9))

        assert not match_old.hit
        assert match_new.hit
        assert cache.stats.evictions >= 1

    def test_lfu_eviction(self) -> None:
        """LFU eviction removes least frequently used."""
        config = CacheConfig(max_entries=2, min_prefix_length=2, eviction_policy="lfu")
        cache = CacheManager(config=config)

        cache.store((1, 2, 3))
        cache.store((4, 5, 6))

        # Access (1,2,3) to bump its frequency
        cache.lookup((1, 2, 3))

        # Storing third entry should evict (4,5,6) which has lower access_count
        cache.store((7, 8, 9))

        match_kept = cache.lookup((1, 2, 3))
        match_evicted = cache.lookup((4, 5, 6))

        assert match_kept.hit
        assert not match_evicted.hit

    def test_manual_eviction(self, small_cache: CacheManager) -> None:
        """Manual eviction removes specific entry."""
        tokens = (1, 2, 3, 4, 5)
        key = small_cache.store(tokens)

        assert small_cache.evict(key) is True
        match = small_cache.lookup(tokens)
        assert not match.hit

    def test_evict_nonexistent(self, small_cache: CacheManager) -> None:
        """Evicting nonexistent key returns False."""
        assert small_cache.evict("nonexistent") is False

    def test_clear_cache(self, small_cache: CacheManager) -> None:
        """Clear removes all entries."""
        small_cache.store((1, 2, 3))
        small_cache.store((4, 5, 6))
        small_cache.clear()

        stats = small_cache.stats
        assert stats.entries_count == 0


class TestCacheManagerTTL:
    """Tests for TTL expiry."""

    def test_expired_entry_returns_miss(self) -> None:
        """Expired entry is treated as a miss."""
        config = CacheConfig(
            default_ttl_seconds=0.1,
            min_prefix_length=2,
        )
        cache = CacheManager(config=config)

        tokens = (1, 2, 3, 4, 5)
        cache.store(tokens)

        # Wait for TTL to expire
        time.sleep(0.15)

        match = cache.lookup(tokens)
        assert not match.hit

    def test_non_expired_entry_hits(self) -> None:
        """Entry within TTL returns hit."""
        config = CacheConfig(
            default_ttl_seconds=10.0,
            min_prefix_length=2,
        )
        cache = CacheManager(config=config)

        tokens = (1, 2, 3, 4, 5)
        cache.store(tokens)

        match = cache.lookup(tokens)
        assert match.hit


class TestCacheManagerBatchAnalysis:
    """Tests for batch prefix analysis."""

    def test_empty_batch(self, small_cache: CacheManager) -> None:
        """Empty batch returns zero analysis."""
        analysis = small_cache.analyze_batch([])
        assert analysis.batch_size == 0
        assert analysis.dedup_ratio == 0.0

    def test_no_shared_prefixes(self, small_cache: CacheManager) -> None:
        """Completely different sequences share nothing."""
        sequences = [
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
        ]
        analysis = small_cache.analyze_batch(sequences)
        assert analysis.batch_size == 3
        assert analysis.unique_prefixes == 0

    def test_shared_prefix_detected(self, small_cache: CacheManager) -> None:
        """Shared prefix is detected in batch analysis."""
        sequences = [
            (1, 2, 3, 4, 5),
            (1, 2, 3, 4, 6),
            (1, 2, 3, 4, 7),
        ]
        analysis = small_cache.analyze_batch(sequences)
        assert analysis.batch_size == 3
        assert analysis.unique_prefixes >= 1
        assert analysis.potential_savings_tokens > 0
