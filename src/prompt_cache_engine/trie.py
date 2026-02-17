"""Radix trie for efficient prefix matching on token sequences."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Estimated bytes per KV cache entry per token per layer
# Typical: 2 (K+V) * hidden_dim * num_layers * dtype_size
DEFAULT_BYTES_PER_TOKEN = 2048


@dataclass
class TrieNode:
    """Node in the radix trie.

    Each node stores a segment of the token sequence (edge label)
    and optionally a cache key if this node marks the end of a cached prefix.

    Args:
        tokens: Token segment stored at this edge
        children: Child nodes keyed by first token of their segment
        cache_key: Key for the cached KV entry at this node (if any)
        depth: Total token depth from root to end of this node's segment
    """

    tokens: Tuple[int, ...] = ()
    children: Dict[int, TrieNode] = field(default_factory=dict)
    cache_key: Optional[str] = None
    depth: int = 0


class RadixTrie:
    """Radix trie (compressed trie) for token sequence prefix matching.

    Stores token sequences efficiently by sharing common prefixes.
    Each edge is labeled with a sequence of tokens, and nodes can
    optionally store a cache key marking a cached prefix boundary.
    """

    def __init__(self) -> None:
        """Initialize an empty radix trie."""
        self.root = TrieNode()
        self._size = 0

    @property
    def size(self) -> int:
        """Number of cached prefix entries in the trie."""
        return self._size

    def insert(
        self,
        tokens: Tuple[int, ...],
        cache_key: str,
    ) -> None:
        """Insert a token sequence with its cache key.

        Args:
            tokens: Token sequence to insert
            cache_key: Key identifying the cached KV entry
        """
        if not tokens:
            return

        node = self.root
        pos = 0

        while pos < len(tokens):
            first_token = tokens[pos]

            if first_token not in node.children:
                # No matching child -- create a new leaf
                new_node = TrieNode(
                    tokens=tokens[pos:],
                    cache_key=cache_key,
                    depth=len(tokens),
                )
                node.children[first_token] = new_node
                self._size += 1
                logger.debug(
                    f"Inserted new leaf: depth={len(tokens)}, key={cache_key}"
                )
                return

            child = node.children[first_token]
            child_tokens = child.tokens

            # Find the longest common prefix between remaining tokens and child edge
            common_len = 0
            while (
                common_len < len(child_tokens)
                and pos + common_len < len(tokens)
                and child_tokens[common_len] == tokens[pos + common_len]
            ):
                common_len += 1

            if common_len == len(child_tokens):
                # Full match on child edge -- descend
                pos += common_len
                node = child
                continue

            # Partial match -- split the edge
            split_node = TrieNode(
                tokens=child_tokens[:common_len],
                depth=node.depth + common_len if hasattr(node, 'depth') else common_len,
            )

            # Move original child under split node
            child.tokens = child_tokens[common_len:]
            split_node.children[child.tokens[0]] = child

            # Add new branch for remaining tokens
            remaining = tokens[pos + common_len:]
            if remaining:
                new_leaf = TrieNode(
                    tokens=remaining,
                    cache_key=cache_key,
                    depth=len(tokens),
                )
                split_node.children[remaining[0]] = new_leaf
            else:
                split_node.cache_key = cache_key

            node.children[first_token] = split_node
            self._size += 1
            logger.debug(
                f"Split edge at common_len={common_len}, key={cache_key}"
            )
            return

        # Reached end of tokens exactly at an existing node
        if node.cache_key is None:
            self._size += 1
        node.cache_key = cache_key

    def find_longest_prefix(
        self,
        tokens: Tuple[int, ...],
    ) -> Tuple[int, Optional[str]]:
        """Find the longest cached prefix matching the given tokens.

        Args:
            tokens: Token sequence to search for

        Returns:
            Tuple of (matched_length, cache_key) where cache_key is None
            if no cached prefix was found
        """
        if not tokens:
            return 0, None

        node = self.root
        pos = 0
        best_length = 0
        best_key: Optional[str] = None

        # Check root for cache key
        if node.cache_key is not None:
            best_length = 0
            best_key = node.cache_key

        while pos < len(tokens):
            first_token = tokens[pos]

            if first_token not in node.children:
                break

            child = node.children[first_token]
            child_tokens = child.tokens

            # Match child edge tokens
            match_len = 0
            while (
                match_len < len(child_tokens)
                and pos + match_len < len(tokens)
                and child_tokens[match_len] == tokens[pos + match_len]
            ):
                match_len += 1

            if match_len < len(child_tokens):
                # Partial match on edge -- can't descend further
                break

            pos += match_len
            node = child

            if node.cache_key is not None:
                best_length = pos
                best_key = node.cache_key

        return best_length, best_key

    def remove(self, tokens: Tuple[int, ...]) -> bool:
        """Remove a cached prefix entry.

        Args:
            tokens: Token sequence to remove

        Returns:
            True if an entry was removed, False if not found
        """
        if not tokens:
            return False

        # Navigate to the node
        node = self.root
        pos = 0
        path: List[Tuple[TrieNode, int]] = []  # (parent, first_token)

        while pos < len(tokens):
            first_token = tokens[pos]
            if first_token not in node.children:
                return False

            path.append((node, first_token))
            child = node.children[first_token]

            match_len = 0
            while (
                match_len < len(child.tokens)
                and pos + match_len < len(tokens)
                and child.tokens[match_len] == tokens[pos + match_len]
            ):
                match_len += 1

            if match_len < len(child.tokens):
                return False

            pos += match_len
            node = child

        if node.cache_key is None:
            return False

        node.cache_key = None
        self._size -= 1
        return True

    def get_all_entries(self) -> List[Tuple[Tuple[int, ...], str]]:
        """Get all cached entries as (token_sequence, cache_key) pairs.

        Returns:
            List of all cached token sequences with their cache keys
        """
        entries: List[Tuple[Tuple[int, ...], str]] = []
        self._collect_entries(self.root, (), entries)
        return entries

    def _collect_entries(
        self,
        node: TrieNode,
        prefix: Tuple[int, ...],
        entries: List[Tuple[Tuple[int, ...], str]],
    ) -> None:
        """Recursively collect all entries from the trie.

        Args:
            node: Current node
            prefix: Token prefix accumulated so far
            entries: List to append entries to
        """
        current = prefix + node.tokens
        if node.cache_key is not None:
            entries.append((current, node.cache_key))
        for child in node.children.values():
            self._collect_entries(child, current, entries)
