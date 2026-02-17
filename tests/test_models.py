"""Tests for data models."""

from __future__ import annotations

import pytest

from prompt_cache_engine.models import BatchAnalysis, CacheConfig, CacheStats, PrefixMatch


class TestCacheConfig:
    """Tests for CacheConfig validation."""

    def test_default_config(self) -> None:
        """Default config has valid defaults."""
        config = CacheConfig()
        assert config.max_entries == 10000
        assert config.max_memory_mb == 1024.0
        assert config.eviction_policy == "lru"
        assert config.min_prefix_length == 4

    def test_custom_config(self) -> None:
        """Custom values are accepted."""
        config = CacheConfig(max_entries=100, max_memory_mb=64.0, eviction_policy="lfu")
        assert config.max_entries == 100
        assert config.eviction_policy == "lfu"

    def test_invalid_max_entries_raises(self) -> None:
        """Zero or negative max_entries raises ValueError."""
        with pytest.raises(ValueError, match="max_entries"):
            CacheConfig(max_entries=0)

    def test_invalid_memory_raises(self) -> None:
        """Non-positive memory raises ValueError."""
        with pytest.raises(ValueError, match="max_memory_mb"):
            CacheConfig(max_memory_mb=-1.0)

    def test_invalid_policy_raises(self) -> None:
        """Unknown eviction policy raises ValueError."""
        with pytest.raises(ValueError, match="eviction_policy"):
            CacheConfig(eviction_policy="fifo")

    def test_invalid_min_prefix_raises(self) -> None:
        """Zero min_prefix_length raises ValueError."""
        with pytest.raises(ValueError, match="min_prefix_length"):
            CacheConfig(min_prefix_length=0)


class TestPrefixMatch:
    """Tests for PrefixMatch."""

    def test_miss_has_zero_savings(self) -> None:
        """A cache miss has zero savings ratio."""
        match = PrefixMatch(total_length=100, hit=False)
        assert match.savings_ratio == 0.0
        assert not match.hit

    def test_full_hit_savings(self) -> None:
        """Full hit gives savings_ratio of 1.0."""
        match = PrefixMatch(
            matched_tokens=(1, 2, 3),
            matched_length=3,
            total_length=3,
            cache_key="abc",
            remaining_tokens=(),
            hit=True,
        )
        assert match.savings_ratio == 1.0

    def test_partial_hit_savings(self) -> None:
        """Partial hit computes correct savings ratio."""
        match = PrefixMatch(
            matched_tokens=(1, 2),
            matched_length=2,
            total_length=10,
            cache_key="abc",
            remaining_tokens=(3, 4, 5, 6, 7, 8, 9, 10),
            hit=True,
        )
        assert match.savings_ratio == pytest.approx(0.2)

    def test_empty_total_length(self) -> None:
        """Zero total length gives zero savings."""
        match = PrefixMatch(total_length=0)
        assert match.savings_ratio == 0.0


class TestCacheStats:
    """Tests for CacheStats."""

    def test_empty_stats(self) -> None:
        """Empty stats have zero rates."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
        assert stats.token_savings_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Hit rate computed correctly."""
        stats = CacheStats(total_lookups=100, cache_hits=75, cache_misses=25)
        assert stats.hit_rate == pytest.approx(0.75)

    def test_token_savings_rate(self) -> None:
        """Token savings rate computed correctly."""
        stats = CacheStats(total_tokens_served=500, total_tokens_requested=1000)
        assert stats.token_savings_rate == pytest.approx(0.5)


class TestBatchAnalysis:
    """Tests for BatchAnalysis."""

    def test_empty_batch(self) -> None:
        """Empty batch has zero dedup ratio."""
        analysis = BatchAnalysis()
        assert analysis.dedup_ratio == 0.0

    @pytest.mark.parametrize(
        "savings,total,expected",
        [
            (50, 100, 0.5),
            (0, 100, 0.0),
            (100, 100, 1.0),
        ],
    )
    def test_dedup_ratio(self, savings: int, total: int, expected: float) -> None:
        """Dedup ratio computed correctly for various inputs."""
        analysis = BatchAnalysis(
            potential_savings_tokens=savings, total_tokens=total
        )
        assert analysis.dedup_ratio == pytest.approx(expected)
