"""Performance benchmarks for prompt-cache-engine components."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

from prompt_cache_engine.cache import CacheManager
from prompt_cache_engine.models import CacheConfig
from prompt_cache_engine.trie import RadixTrie

logging.basicConfig(level=logging.WARNING)

NUM_ITERATIONS = 3


@dataclass
class BenchResult:
    """Benchmark result."""

    name: str
    mean_seconds: float
    items_processed: int
    throughput_per_sec: float


def generate_token_sequences(
    count: int,
    prefix_length: int,
    suffix_length: int,
    num_unique_prefixes: int = 10,
) -> List[Tuple[int, ...]]:
    """Generate token sequences with shared prefixes.

    Args:
        count: Number of sequences to generate
        prefix_length: Length of shared prefix in tokens
        suffix_length: Length of unique suffix in tokens
        num_unique_prefixes: Number of distinct prefixes

    Returns:
        List of token sequences
    """
    sequences: List[Tuple[int, ...]] = []
    for i in range(count):
        prefix_id = i % num_unique_prefixes
        prefix = tuple(range(prefix_id * 1000, prefix_id * 1000 + prefix_length))
        suffix = tuple(range(i * 10000, i * 10000 + suffix_length))
        sequences.append(prefix + suffix)
    return sequences


def bench_trie_insert() -> BenchResult:
    """Benchmark trie insertion speed."""
    n_entries = 50000
    sequences = generate_token_sequences(n_entries, prefix_length=20, suffix_length=10)

    timings: List[float] = []
    for _ in range(NUM_ITERATIONS):
        trie = RadixTrie()
        start = time.perf_counter()
        for i, seq in enumerate(sequences):
            trie.insert(seq, f"key-{i}")
        timings.append(time.perf_counter() - start)

    mean_time = sum(timings) / len(timings)
    return BenchResult(
        name=f"Trie Insert ({n_entries} entries)",
        mean_seconds=mean_time,
        items_processed=n_entries,
        throughput_per_sec=n_entries / mean_time,
    )


def bench_trie_lookup() -> BenchResult:
    """Benchmark trie prefix lookup speed."""
    n_entries = 10000
    sequences = generate_token_sequences(n_entries, prefix_length=20, suffix_length=10)

    trie = RadixTrie()
    for i, seq in enumerate(sequences):
        trie.insert(seq, f"key-{i}")

    # Generate lookup queries (mix of hits and misses)
    queries = sequences[:5000] + generate_token_sequences(5000, prefix_length=20, suffix_length=15)
    n_lookups = len(queries)

    timings: List[float] = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        for query in queries:
            trie.find_longest_prefix(query)
        timings.append(time.perf_counter() - start)

    mean_time = sum(timings) / len(timings)
    return BenchResult(
        name=f"Trie Lookup ({n_lookups} queries)",
        mean_seconds=mean_time,
        items_processed=n_lookups,
        throughput_per_sec=n_lookups / mean_time,
    )


def bench_cache_store() -> BenchResult:
    """Benchmark cache store operations."""
    n_entries = 10000
    sequences = generate_token_sequences(n_entries, prefix_length=20, suffix_length=10)

    config = CacheConfig(max_entries=n_entries + 1, min_prefix_length=2)

    timings: List[float] = []
    for _ in range(NUM_ITERATIONS):
        cache = CacheManager(config=config)
        start = time.perf_counter()
        for seq in sequences:
            cache.store(seq, memory_bytes=1024)
        timings.append(time.perf_counter() - start)

    mean_time = sum(timings) / len(timings)
    return BenchResult(
        name=f"Cache Store ({n_entries} entries)",
        mean_seconds=mean_time,
        items_processed=n_entries,
        throughput_per_sec=n_entries / mean_time,
    )


def bench_cache_lookup() -> BenchResult:
    """Benchmark cache lookup with prefix matching."""
    n_entries = 5000
    sequences = generate_token_sequences(n_entries, prefix_length=20, suffix_length=10)

    config = CacheConfig(max_entries=n_entries + 1, min_prefix_length=2)
    cache = CacheManager(config=config)
    for seq in sequences:
        cache.store(seq, memory_bytes=1024)

    # Mix of exact hits and prefix hits
    queries = sequences[:2500]
    # Add queries with same prefix but different suffix
    for seq in sequences[2500:5000]:
        queries.append(seq[:20] + (99999,) * 15)  # Same prefix, different suffix

    n_lookups = len(queries)

    timings: List[float] = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        for query in queries:
            cache.lookup(query)
        timings.append(time.perf_counter() - start)

    mean_time = sum(timings) / len(timings)
    return BenchResult(
        name=f"Cache Lookup ({n_lookups} queries)",
        mean_seconds=mean_time,
        items_processed=n_lookups,
        throughput_per_sec=n_lookups / mean_time,
    )


def bench_batch_analysis() -> BenchResult:
    """Benchmark batch prefix analysis."""
    n_sequences = 1000
    sequences = generate_token_sequences(n_sequences, prefix_length=50, suffix_length=20)

    config = CacheConfig(min_prefix_length=4)
    cache = CacheManager(config=config)

    timings: List[float] = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        cache.analyze_batch(sequences)
        timings.append(time.perf_counter() - start)

    mean_time = sum(timings) / len(timings)
    return BenchResult(
        name=f"Batch Analysis ({n_sequences} sequences)",
        mean_seconds=mean_time,
        items_processed=n_sequences,
        throughput_per_sec=n_sequences / mean_time,
    )


def main() -> None:
    """Run all benchmarks."""
    benchmarks = [
        bench_trie_insert,
        bench_trie_lookup,
        bench_cache_store,
        bench_cache_lookup,
        bench_batch_analysis,
    ]

    results: List[BenchResult] = []
    for bench_fn in benchmarks:
        results.append(bench_fn())

    header = f"{'Benchmark':<50} {'Time (s)':>10} {'Items':>8} {'Throughput':>14}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("Prompt Cache Engine - Performance Benchmarks")
    print("=" * len(header))
    print()
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r.name:<50} {r.mean_seconds:>10.4f} {r.items_processed:>8} "
            f"{r.throughput_per_sec:>12.2f}/s"
        )

    print(sep)
    print()


if __name__ == "__main__":
    main()
