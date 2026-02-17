"""Tests for radix trie implementation."""

from __future__ import annotations

import pytest

from prompt_cache_engine.trie import RadixTrie


class TestRadixTrieInsertAndLookup:
    """Tests for insert and prefix lookup."""

    def test_empty_trie_returns_no_match(self, trie: RadixTrie) -> None:
        """Empty trie returns zero-length match."""
        length, key = trie.find_longest_prefix((1, 2, 3))
        assert length == 0
        assert key is None

    def test_exact_match(self, trie: RadixTrie) -> None:
        """Exact token match returns correct key."""
        tokens = (1, 2, 3, 4)
        trie.insert(tokens, "key-1234")
        length, key = trie.find_longest_prefix(tokens)
        assert length == 4
        assert key == "key-1234"

    def test_prefix_match(self, trie: RadixTrie) -> None:
        """Cached prefix matches longer query."""
        trie.insert((1, 2, 3), "key-123")
        length, key = trie.find_longest_prefix((1, 2, 3, 4, 5))
        assert length == 3
        assert key == "key-123"

    def test_no_match_for_different_prefix(self, trie: RadixTrie) -> None:
        """Unrelated tokens return no match."""
        trie.insert((1, 2, 3), "key-123")
        length, key = trie.find_longest_prefix((4, 5, 6))
        assert length == 0
        assert key is None

    def test_longest_prefix_wins(self, trie: RadixTrie) -> None:
        """When multiple prefixes match, longest wins."""
        trie.insert((1, 2), "key-12")
        trie.insert((1, 2, 3, 4), "key-1234")
        length, key = trie.find_longest_prefix((1, 2, 3, 4, 5, 6))
        assert length == 4
        assert key == "key-1234"

    def test_shorter_prefix_when_longer_doesnt_match(self, trie: RadixTrie) -> None:
        """Falls back to shorter prefix when longer doesn't fully match."""
        trie.insert((1, 2), "key-12")
        trie.insert((1, 2, 3, 4), "key-1234")
        length, key = trie.find_longest_prefix((1, 2, 5, 6))
        assert length == 2
        assert key == "key-12"

    def test_multiple_branches(self, trie: RadixTrie) -> None:
        """Trie correctly handles branching paths."""
        trie.insert((1, 2, 3), "key-a")
        trie.insert((1, 2, 4), "key-b")
        trie.insert((1, 5, 6), "key-c")

        _, key_a = trie.find_longest_prefix((1, 2, 3, 7))
        _, key_b = trie.find_longest_prefix((1, 2, 4, 8))
        _, key_c = trie.find_longest_prefix((1, 5, 6, 9))

        assert key_a == "key-a"
        assert key_b == "key-b"
        assert key_c == "key-c"

    def test_empty_tokens_insert(self, trie: RadixTrie) -> None:
        """Empty token sequence is a no-op."""
        trie.insert((), "key-empty")
        assert trie.size == 0

    def test_empty_tokens_lookup(self, trie: RadixTrie) -> None:
        """Empty token lookup returns no match."""
        length, key = trie.find_longest_prefix(())
        assert length == 0
        assert key is None

    def test_size_tracking(self, trie: RadixTrie) -> None:
        """Size correctly tracks number of entries."""
        assert trie.size == 0
        trie.insert((1, 2, 3), "a")
        assert trie.size == 1
        trie.insert((1, 2, 4), "b")
        assert trie.size == 2
        trie.insert((5, 6), "c")
        assert trie.size == 3


class TestRadixTrieRemove:
    """Tests for entry removal."""

    def test_remove_existing(self, trie: RadixTrie) -> None:
        """Removing existing entry returns True and removes it."""
        trie.insert((1, 2, 3), "key-123")
        assert trie.remove((1, 2, 3)) is True
        assert trie.size == 0
        length, key = trie.find_longest_prefix((1, 2, 3))
        assert key is None

    def test_remove_nonexistent(self, trie: RadixTrie) -> None:
        """Removing nonexistent entry returns False."""
        assert trie.remove((1, 2, 3)) is False

    def test_remove_preserves_siblings(self, trie: RadixTrie) -> None:
        """Removing one entry preserves sibling entries."""
        trie.insert((1, 2, 3), "a")
        trie.insert((1, 2, 4), "b")
        trie.remove((1, 2, 3))
        _, key = trie.find_longest_prefix((1, 2, 4))
        assert key == "b"

    def test_remove_empty_tokens(self, trie: RadixTrie) -> None:
        """Removing empty tokens returns False."""
        assert trie.remove(()) is False


class TestRadixTrieGetAllEntries:
    """Tests for collecting all entries."""

    def test_empty_trie(self, trie: RadixTrie) -> None:
        """Empty trie returns no entries."""
        assert trie.get_all_entries() == []

    def test_all_entries_returned(self, trie: RadixTrie) -> None:
        """All inserted entries are returned."""
        trie.insert((1, 2, 3), "a")
        trie.insert((1, 2, 4), "b")
        trie.insert((5, 6), "c")

        entries = trie.get_all_entries()
        keys = {key for _, key in entries}
        assert keys == {"a", "b", "c"}

    @pytest.mark.parametrize(
        "tokens,expected_key",
        [
            ((10, 20, 30), "p1"),
            ((10, 20, 40), "p2"),
            ((50, 60), "p3"),
        ],
    )
    def test_parametrized_insert_and_find(
        self,
        trie: RadixTrie,
        tokens: tuple,
        expected_key: str,
    ) -> None:
        """Parametrized test for insert and lookup."""
        trie.insert(tokens, expected_key)
        length, key = trie.find_longest_prefix(tokens)
        assert length == len(tokens)
        assert key == expected_key
