"""Tests for utility functions."""

from __future__ import annotations

import pytest

from prompt_cache_engine.models import CacheStats
from prompt_cache_engine.utils import (
    find_common_prefix_length,
    format_stats_report,
    tokenize_simple,
)


class TestTokenizeSimple:
    """Tests for simple tokenizer."""

    def test_basic_tokenization(self) -> None:
        """Words are tokenized to integer tuples."""
        tokens = tokenize_simple("hello world")
        assert isinstance(tokens, tuple)
        assert len(tokens) == 2
        assert all(isinstance(t, int) for t in tokens)

    def test_deterministic(self) -> None:
        """Same text produces same tokens."""
        assert tokenize_simple("hello world") == tokenize_simple("hello world")

    def test_empty_string(self) -> None:
        """Empty string produces empty tuple."""
        assert tokenize_simple("") == ()

    def test_single_word(self) -> None:
        """Single word produces single token."""
        tokens = tokenize_simple("hello")
        assert len(tokens) == 1


class TestFindCommonPrefixLength:
    """Tests for common prefix detection."""

    def test_identical_sequences(self) -> None:
        """Identical sequences return full length."""
        assert find_common_prefix_length((1, 2, 3), (1, 2, 3)) == 3

    def test_no_common_prefix(self) -> None:
        """Completely different sequences return 0."""
        assert find_common_prefix_length((1, 2, 3), (4, 5, 6)) == 0

    def test_partial_prefix(self) -> None:
        """Partial prefix returns correct length."""
        assert find_common_prefix_length((1, 2, 3, 4), (1, 2, 5, 6)) == 2

    def test_empty_sequence(self) -> None:
        """Empty sequence returns 0."""
        assert find_common_prefix_length((), (1, 2, 3)) == 0

    @pytest.mark.parametrize(
        "seq_a,seq_b,expected",
        [
            ((1,), (1, 2, 3), 1),
            ((1, 2, 3), (1,), 1),
            ((1, 2), (1, 2), 2),
        ],
    )
    def test_different_lengths(
        self,
        seq_a: tuple,
        seq_b: tuple,
        expected: int,
    ) -> None:
        """Sequences of different lengths handled correctly."""
        assert find_common_prefix_length(seq_a, seq_b) == expected


class TestFormatStatsReport:
    """Tests for stats report formatting."""

    def test_report_contains_key_fields(self) -> None:
        """Report includes all key metrics."""
        stats = CacheStats(
            total_lookups=100,
            cache_hits=75,
            cache_misses=25,
            entries_count=10,
            memory_used_mb=5.5,
        )
        report = format_stats_report(stats)
        assert "100" in report
        assert "75" in report
        assert "75.0%" in report
        assert "5.50" in report
