"""Shared test fixtures for prompt-cache-engine."""

from __future__ import annotations

import pytest

from prompt_cache_engine.cache import CacheManager
from prompt_cache_engine.models import CacheConfig
from prompt_cache_engine.trie import RadixTrie


@pytest.fixture
def trie() -> RadixTrie:
    """Empty radix trie."""
    return RadixTrie()


@pytest.fixture
def small_cache() -> CacheManager:
    """Cache manager with small limits for testing."""
    config = CacheConfig(
        max_entries=10,
        max_memory_mb=1.0,
        min_prefix_length=2,
    )
    return CacheManager(config=config)


@pytest.fixture
def default_cache() -> CacheManager:
    """Cache manager with default config."""
    return CacheManager()


@pytest.fixture
def sample_tokens() -> tuple:
    """Sample token sequences for testing."""
    return (
        (1, 2, 3, 4, 5, 6, 7, 8),
        (1, 2, 3, 4, 9, 10, 11, 12),
        (1, 2, 3, 4, 5, 6, 13, 14),
        (20, 21, 22, 23, 24, 25),
    )
