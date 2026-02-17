"""Quick start example: Demonstrate prefix caching with shared system prompts."""

from __future__ import annotations

import logging

from prompt_cache_engine.cache import CacheManager
from prompt_cache_engine.models import CacheConfig
from prompt_cache_engine.utils import format_batch_analysis, format_stats_report, tokenize_simple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    """Run prefix caching demonstration."""
    print("=" * 70)
    print("Prompt Cache Engine - Quick Start Example")
    print("=" * 70)
    print()

    # Configure cache with small limits for demonstration
    config = CacheConfig(
        max_entries=100,
        max_memory_mb=10.0,
        min_prefix_length=3,
    )
    cache = CacheManager(config=config)

    # Simulate prompts that share a common system prompt prefix
    system_prompt = (
        "You are a helpful AI assistant specializing in Python programming. "
        "Always provide clear explanations with code examples. "
        "Follow best practices and include error handling."
    )

    user_queries = [
        "How do I read a CSV file in Python?",
        "How do I sort a list of dictionaries by a key?",
        "How do I handle exceptions in async code?",
        "How do I implement a binary search tree?",
        "How do I read a JSON file and parse it?",
    ]

    # Build full prompts (system + user query)
    full_prompts = [f"{system_prompt} {query}" for query in user_queries]

    # Tokenize all prompts
    token_sequences = [tokenize_simple(p) for p in full_prompts]

    # Phase 1: Analyze batch prefix sharing potential
    print("-" * 70)
    print("Phase 1: Batch Prefix Analysis")
    print("-" * 70)
    print()
    analysis = cache.analyze_batch(token_sequences)
    print(format_batch_analysis(analysis))
    print()

    # Phase 2: Process prompts through cache
    print("-" * 70)
    print("Phase 2: Sequential Cache Processing")
    print("-" * 70)
    print()

    for i, (prompt, tokens) in enumerate(zip(full_prompts, token_sequences)):
        match = cache.lookup(tokens)

        if match.hit:
            print(
                f"  [{i+1}] HIT  - {match.matched_length}/{match.total_length} tokens cached "
                f"({match.savings_ratio:.0%} savings)"
            )
            print(f"        Query: {user_queries[i][:60]}")
        else:
            cache.store(tokens)
            print(
                f"  [{i+1}] MISS - Stored {len(tokens)} tokens"
            )
            print(f"        Query: {user_queries[i][:60]}")

    print()

    # Phase 3: Show cache statistics
    print("-" * 70)
    print("Phase 3: Cache Statistics")
    print("-" * 70)
    print()
    print(format_stats_report(cache.stats))
    print()

    # Phase 4: Demonstrate second pass (all hits)
    print("-" * 70)
    print("Phase 4: Second Pass (all cached)")
    print("-" * 70)
    print()

    for i, tokens in enumerate(token_sequences):
        match = cache.lookup(tokens)
        status = "HIT" if match.hit else "MISS"
        print(f"  [{i+1}] {status} - {user_queries[i][:60]}")

    print()
    print(format_stats_report(cache.stats))
    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
