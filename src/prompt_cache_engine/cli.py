"""Command-line interface for prompt-cache-engine."""

from __future__ import annotations

import logging
import sys

import click

from prompt_cache_engine.cache import CacheManager
from prompt_cache_engine.models import CacheConfig
from prompt_cache_engine.utils import (
    format_batch_analysis,
    format_stats_report,
    tokenize_simple,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Prompt Cache Engine - KV cache sharing for prompt prefix deduplication."""


@main.command()
@click.argument("prompts", nargs=-1, required=True)
@click.option("--min-prefix", default=4, help="Minimum prefix length in tokens")
def analyze(prompts: tuple, min_prefix: int) -> None:
    """Analyze prefix sharing potential across prompts."""
    config = CacheConfig(min_prefix_length=min_prefix)
    manager = CacheManager(config=config)

    token_sequences = [tokenize_simple(p) for p in prompts]
    analysis = manager.analyze_batch(token_sequences)
    click.echo(format_batch_analysis(analysis))


@main.command()
@click.argument("prompts", nargs=-1, required=True)
@click.option("--max-entries", default=1000, help="Maximum cache entries")
@click.option("--min-prefix", default=4, help="Minimum prefix length")
def demo(prompts: tuple, max_entries: int, min_prefix: int) -> None:
    """Run a demo of the cache engine with the given prompts."""
    config = CacheConfig(max_entries=max_entries, min_prefix_length=min_prefix)
    manager = CacheManager(config=config)

    click.echo(f"Processing {len(prompts)} prompts...\n")

    for prompt in prompts:
        tokens = tokenize_simple(prompt)
        match = manager.lookup(tokens)

        if match.hit:
            click.echo(f"HIT:  '{prompt[:50]}...' ({match.matched_length}/{match.total_length} tokens cached)")
        else:
            manager.store(tokens)
            click.echo(f"MISS: '{prompt[:50]}...' ({len(tokens)} tokens stored)")

    click.echo()
    click.echo(format_stats_report(manager.stats))


if __name__ == "__main__":
    main()
