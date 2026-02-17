"""Utility functions for prompt-cache-engine."""

from __future__ import annotations

import logging
from typing import List, Tuple

from prompt_cache_engine.models import BatchAnalysis, CacheStats

logger = logging.getLogger(__name__)


def format_stats_report(stats: CacheStats) -> str:
    """Format cache statistics as a human-readable report.

    Args:
        stats: CacheStats to format

    Returns:
        Formatted report string
    """
    lines = [
        "=== Prompt Cache Engine Statistics ===",
        f"Entries:        {stats.entries_count}",
        f"Memory Used:    {stats.memory_used_mb:.2f} MB",
        f"Total Lookups:  {stats.total_lookups}",
        f"Cache Hits:     {stats.cache_hits}",
        f"Cache Misses:   {stats.cache_misses}",
        f"Hit Rate:       {stats.hit_rate:.1%}",
        f"Tokens Served:  {stats.total_tokens_served}",
        f"Tokens Requested: {stats.total_tokens_requested}",
        f"Token Savings:  {stats.token_savings_rate:.1%}",
        f"Evictions:      {stats.evictions}",
        "=====================================",
    ]
    return "\n".join(lines)


def format_batch_analysis(analysis: BatchAnalysis) -> str:
    """Format batch analysis as a human-readable report.

    Args:
        analysis: BatchAnalysis to format

    Returns:
        Formatted report string
    """
    lines = [
        "=== Batch Prefix Analysis ===",
        f"Batch Size:       {analysis.batch_size}",
        f"Unique Prefixes:  {analysis.unique_prefixes}",
        f"Total Tokens:     {analysis.total_tokens}",
        f"Saveable Tokens:  {analysis.potential_savings_tokens}",
        f"Dedup Ratio:      {analysis.dedup_ratio:.1%}",
    ]

    if analysis.shared_prefix_groups:
        lines.append("Shared Groups:")
        for key, indices in analysis.shared_prefix_groups.items():
            lines.append(f"  {key}: {len(indices)} prompts")

    lines.append("==============================")
    return "\n".join(lines)


def tokenize_simple(text: str) -> Tuple[int, ...]:
    """Simple whitespace tokenizer for demonstration purposes.

    In production, use a real tokenizer (tiktoken, sentencepiece, etc.).

    Args:
        text: Text to tokenize

    Returns:
        Tuple of token IDs (hash-based)
    """
    words = text.split()
    return tuple(hash(w) % 100000 for w in words)


def find_common_prefix_length(
    seq_a: Tuple[int, ...],
    seq_b: Tuple[int, ...],
) -> int:
    """Find the length of the common prefix between two sequences.

    Args:
        seq_a: First token sequence
        seq_b: Second token sequence

    Returns:
        Length of common prefix
    """
    length = min(len(seq_a), len(seq_b))
    for i in range(length):
        if seq_a[i] != seq_b[i]:
            return i
    return length
