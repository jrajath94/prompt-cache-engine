# Architecture: prompt-cache-engine

## Overview

prompt-cache-engine is a standalone prefix caching library for LLM inference workloads. It provides two core primitives: a radix trie for efficient longest-prefix matching on token sequences, and a cache manager with LRU/LFU eviction and TTL expiry. The library is engine-agnostic -- it manages cache keys and metadata, not the actual KV tensors.

## Component Responsibilities

### Radix Trie (`trie.py`)

- **TrieNode**: Stores a token segment (edge label) and optionally a cache key marking a cached boundary
- **RadixTrie**: Compressed trie that shares common token prefixes. Operations:
  - `insert(tokens, cache_key)`: Add a token sequence with associated cache key; splits edges on partial matches
  - `find_longest_prefix(tokens)`: Return the longest cached prefix matching the query; O(L) where L = query length
  - `remove(tokens)`: Remove a cached entry by its token sequence
  - `get_all_entries()`: Recursive collection for debugging/export

### Cache Manager (`cache.py`)

- **CacheEntry**: Metadata for a single cached KV state -- tokens, opaque kv_data, memory estimate, access tracking
- **CacheManager**: Orchestrates trie lookups with an `OrderedDict` for LRU ordering:
  - `lookup(tokens)`: Prefix search + access tracking + TTL check; returns `PrefixMatch`
  - `store(tokens, kv_data)`: Insert + auto-eviction if over capacity
  - `analyze_batch(sequences)`: Pre-flight analysis of prefix sharing across a batch
  - `evict(key)` / `clear()`: Manual cache management

### Models (`models.py`)

- **CacheConfig**: Validated configuration with max_entries, max_memory_mb, eviction_policy, TTL, min_prefix_length
- **PrefixMatch**: Lookup result with matched/remaining tokens, savings ratio
- **CacheStats**: Aggregated metrics -- hit rate, token savings rate, eviction count
- **BatchAnalysis**: Deduplication potential analysis for prompt batches

## Data Flow

```
Token Sequence
    |
    v
CacheManager.lookup()
    |
    v
RadixTrie.find_longest_prefix()  -- O(L) traversal
    |
    v
Check CacheEntry exists + not expired
    |
    v
Update access tracking (last_accessed, access_count, move_to_end)
    |
    v
Return PrefixMatch (hit/miss, matched tokens, remaining tokens)

On Store:
    Token Sequence + KV Data
        |
        v
    _ensure_capacity()  -- evict LRU/LFU entries until space available
        |
        v
    RadixTrie.insert() + OrderedDict insertion
```

## Key Design Choices

1. **Radix trie vs hash table**: Hash tables require exact-length keys. Radix tries support longest-prefix matching naturally, which is the core operation for KV cache reuse.

2. **Engine-agnostic**: KV data is stored as `Any`. The cache manages keys, metadata, and eviction -- the caller is responsible for the actual tensor data. This allows the same cache manager to work with PyTorch tensors, numpy arrays, or even references to GPU memory.

3. **SHA-256 truncated keys**: Content-addressable cache keys mean the same token sequence always maps to the same entry, regardless of when it was cached. Truncation to 16 hex chars (64 bits) provides adequate collision resistance for cache sizes up to millions of entries.

4. **Dual eviction (TTL + capacity)**: TTL prevents serving stale KV states when model weights change. Capacity-based LRU/LFU handles the common case of bounded memory.
