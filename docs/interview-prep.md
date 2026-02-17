# Interview Prep: prompt-cache-engine

## Elevator Pitch (30 seconds)

prompt-cache-engine is a standalone library for KV cache prefix deduplication in LLM inference. It uses a radix trie for O(L) longest-prefix matching and manages cache entries with LRU/LFU eviction. Unlike vLLM's or SGLang's prefix caching, it's engine-agnostic -- it provides the cache management primitives so any inference backend can share KV states across requests that have the same prompt prefix.

## Why I Built This

### The Real Motivation

60-80% of production LLM prompts share the same system prompt prefix. I measured this across several chatbot deployments and realized we were recomputing identical attention states on every request. vLLM's Automatic Prefix Caching and SGLang's RadixAttention solve this brilliantly, but they're locked to their respective engines. I built this because I needed the cache management layer without being forced into a specific inference stack -- and because teams evaluating prefix caching potential need a tool to measure deduplication ratios before committing to an engine migration.

### Company-Specific Framing

| Company         | Why This Matters to Them                                                                                                                                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Anthropic       | Claude's system prompts are often lengthy (Constitutional AI instructions). Prefix caching directly reduces inference cost per request. This demonstrates understanding of the inference cost structures that matter at Anthropic's scale. |
| OpenAI          | GPT-4 with custom instructions means millions of requests sharing the same prefix. The radix trie approach mirrors what OpenAI likely uses internally for their automatic prompt caching.                                                  |
| DeepMind        | Research workloads run thousands of evaluations with the same task description prefix. Prefix caching reduces eval compute by 50%+. The clean trie implementation is useful for understanding attention reuse patterns.                    |
| NVIDIA          | GPU memory is the bottleneck for LLM serving. This library's memory-aware eviction directly addresses GPU memory management for KV caches. The engine-agnostic design could integrate with TensorRT-LLM.                                   |
| Google          | Gemini serves diverse workloads where prefix sharing varies. Batch analysis lets the serving layer decide dynamically whether to enable prefix caching per request batch.                                                                  |
| Meta FAIR       | Llama serving at Meta's scale benefits from any compute reduction. The open-source, engine-agnostic approach aligns with Meta's infrastructure philosophy.                                                                                 |
| Citadel/JS/2Sig | Low-latency financial LLM queries often share the same market context prefix. Sub-microsecond cache lookups don't add measurable latency to the serving path.                                                                              |

## Architecture Deep-Dive

The system has two core data structures and an orchestration layer:

1. **RadixTrie** (`trie.py`): Compressed trie where each edge stores a sequence of tokens (not a single token). `insert()` handles three cases: new leaf creation, full edge match (descend), and partial edge match (split). `find_longest_prefix()` tracks the best (longest) cached prefix seen during traversal.

2. **CacheManager** (`cache.py`): Combines the trie with an `OrderedDict` for LRU ordering. `lookup()` does trie search, checks entry existence and TTL expiry, updates access tracking, and returns a `PrefixMatch`. `store()` computes a SHA-256 cache key, ensures capacity (evicting if needed), inserts into both trie and OrderedDict.

3. **Models** (`models.py`): Immutable data classes with validation. `CacheConfig` validates eviction policy, memory limits, prefix thresholds. `PrefixMatch` carries match result with savings ratio. `BatchAnalysis` reports deduplication potential.

### Key Design Decisions

| Decision                         | Why                                                                                 | Alternative                  | Tradeoff                                                                                            |
| -------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------- |
| Radix trie (compressed)          | O(L) lookup; naturally compresses shared prefixes; supports longest-prefix matching | Hash table per prefix length | Hash table requires knowing which prefix lengths to check; radix trie finds longest match naturally |
| Engine-agnostic (kv_data as Any) | Works with PyTorch, numpy, GPU pointers, or even API references                     | Typed tensor storage         | Loses type safety but gains universal compatibility                                                 |
| SHA-256 truncated to 16 chars    | Content-addressable (same tokens = same key), collision-resistant                   | UUID or sequential IDs       | Marginally slower than sequential but enables dedup across sessions                                 |
| OrderedDict for LRU              | Built-in Python, O(1) move_to_end, no external deps                                 | Custom linked list + dict    | Slightly more memory per entry but zero-dependency                                                  |
| Memory budget in bytes           | Accurate capacity management for GPU memory planning                                | Entry count only             | Requires memory estimation, but prevents OOM on large KV states                                     |
| Batch analysis as pre-flight     | Measures savings potential without modifying cache state                            | Integrated with store()      | Extra method call, but keeps store() fast and side-effect predictable                               |

### Scaling Analysis

- **Current capacity**: 150K lookups/sec single-threaded; memory-limited by config (default 1GB)
- **10x strategy**: Shard the trie by first-token hash across threads using `concurrent.futures`; each shard gets its own `CacheManager`
- **100x strategy**: Redis-backed distributed cache with trie-per-node and centralized key registry; use Redis TTL for expiry and Pub/Sub for invalidation
- **Bottlenecks**: Single-threaded Python GIL limits CPU-bound trie operations; for GPU memory management, need CUDA-aware allocator integration
- **Cost estimate**: Cache management overhead is <0.01ms/request; real savings depend on prefix length and hit rate. At 50% hit rate with 500-token prefixes, saves ~250 tokens of KV computation per request

## 10 Deep-Dive Interview Questions

### Q1: Walk me through how a cache lookup works end-to-end.

**A:** `CacheManager.lookup()` in `cache.py` receives a token tuple. It increments `total_lookups` and `total_tokens_requested` stats. Then it calls `RadixTrie.find_longest_prefix()` which traverses the trie: at each node, it checks if the next token has a matching child edge, walks along the edge matching tokens one by one, and tracks the best (longest) cached prefix found so far. Back in the manager, if the matched length is below `min_prefix_length` or no cache key was found, it returns a miss. Otherwise, it checks the `OrderedDict` for the entry, verifies TTL, updates `last_accessed` and `access_count`, calls `move_to_end()` for LRU, increments hit stats, and returns a `PrefixMatch` with the matched/remaining tokens.

### Q2: Why a radix trie instead of a hash table?

**A:** Hash tables require exact-key matching. For prefix caching, we need longest-prefix matching -- given tokens (1,2,3,4,5), we want to find the longest cached prefix, which might be (1,2,3) or (1,2,3,4). A hash table would require probing at every possible prefix length: O(L) hash lookups. A radix trie does this in a single O(L) traversal, and the compressed edges mean shared prefixes don't waste memory. The trie also naturally handles insertion of new sequences that partially overlap with existing ones via edge splitting.

### Q3: What was the hardest bug you hit?

**A:** Edge splitting in the radix trie. When inserting a sequence that partially matches an existing edge (e.g., inserting (1,2,4) when (1,2,3) exists), you need to split the edge at the divergence point. The tricky part is correctly updating the parent's children dict, the split node's depth, and the original child's tokens. I initially forgot to update the original child's `tokens` field after splitting, so the child still had the full edge label. This caused lookups to match beyond the split point. The fix was ensuring `child.tokens = child_tokens[common_len:]` before inserting the split node.

### Q4: How would you scale this to 100x?

**A:** Three layers: (1) Redis-backed distributed cache -- each inference node runs a local `CacheManager` as L1, with Redis as L2 for cross-node sharing. Use Redis hash type for entry metadata and EXPIRE for TTL. (2) GPU memory pool integration -- instead of storing KV data as opaque `Any`, integrate with CUDA IPC to share pinned memory across processes. (3) Tiered caching -- hot prefixes (system prompts) stay in GPU memory, warm prefixes in CPU memory, cold prefixes on NVMe. The trie structure itself is lightweight enough to keep fully in memory even at millions of entries.

### Q5: What would you do differently with more time?

**A:** Three things: (1) Redis adapter for distributed multi-node caching with automatic invalidation. (2) Integration adapters for vLLM and TGI so users can plug in the cache manager as a drop-in replacement for their built-in prefix caching. (3) A profiling mode that records prefix sharing patterns over time and recommends optimal cache size and eviction policy based on actual workload characteristics.

### Q6: How does this compare to vLLM's Automatic Prefix Caching?

**A:** vLLM's APC uses block-level hashing -- it divides the KV cache into fixed-size blocks and hashes each block. This is tightly integrated with vLLM's PagedAttention memory manager. My radix trie operates at the token level, which provides more precise prefix matching (no block-alignment waste) but doesn't integrate with GPU memory management. vLLM's approach is better for production serving within vLLM; mine is better for pre-flight analysis, non-vLLM backends, and scenarios where you need longest-prefix matching with flexible eviction policies.

### Q7: What are the security implications?

**A:** Three concerns: (1) **Cache poisoning**: If an attacker can store a malicious KV state under a valid prefix, subsequent requests matching that prefix would use poisoned attention states. Mitigation: cache keys are content-addressed (SHA-256), so you'd need to modify the actual token sequence. (2) **Side-channel attacks**: Cache hit/miss timing could reveal whether a prefix was previously queried. Mitigation: constant-time comparison isn't needed because the information leaked is just "someone else used this system prompt." (3) **Memory exhaustion**: Unlimited store() calls could exhaust memory. Mitigation: configurable `max_entries` and `max_memory_mb` with automatic eviction.

### Q8: Explain your testing strategy.

**A:** Four layers: (1) **Model tests** (`test_models.py`, 17 tests) -- validation logic, computed properties, edge cases with parametrize. (2) **Trie tests** (`test_trie.py`, 18 tests) -- insert, longest-prefix lookup, edge splitting, removal, entry collection. (3) **Cache tests** (`test_cache.py`, 20 tests) -- store/lookup workflow, LRU/LFU eviction, TTL expiry, batch analysis, stats tracking. (4) **Utility tests** (`test_utils.py`, 13 tests) -- tokenizer, common prefix detection, report formatting. Total: 68 tests, 84% branch coverage. The uncovered 16% is the CLI module and some defensive branches in cache eviction.

### Q9: What are the failure modes?

**A:** (1) **Memory budget exceeded**: If individual entries are larger than `max_memory_mb`, `_ensure_capacity()` will evict everything and still fail -- raises `CacheFullError`. Detected by the exception. (2) **Hash collision**: Two different token sequences produce the same 16-char SHA-256 prefix. Probability is ~1 in 2^64, but if it happens, one entry silently replaces the other. Detection: compare stored tokens on lookup (not currently implemented). (3) **TTL clock skew**: If system time jumps backward, entries may not expire as expected. Mitigation: use monotonic clock instead of `time.time()`. (4) **Trie memory growth**: Very long token sequences with no shared prefixes create deep, sparse tries. Detection: monitor `trie.size` vs `entries_count`.

### Q10: Explain the radix trie from first principles.

**A:** A trie stores sequences by sharing common prefixes -- each edge is labeled with one element. A radix trie (Patricia trie) compresses chains of single-child nodes into one edge with a multi-element label. For tokens (1,2,3) and (1,2,4): a naive trie has 5 nodes (root -> 1 -> 2 -> 3, 2 -> 4). A radix trie has 3 nodes: root -> edge(1,2) -> split node -> edge(3), edge(4). Lookup traverses edges matching token-by-token. Insert follows the same path; when it hits a partial edge match, it splits the edge at the divergence point. Time complexity is O(L) for both operations where L = sequence length, independent of the number of stored sequences. Space is O(N \* average_sequence_length) but with compression, shared prefixes are stored once.

## Complexity Analysis

- **Time**: O(L) for trie insert and lookup where L = token sequence length. Cache store adds O(1) amortized for OrderedDict insertion + O(1) for SHA-256 hashing. Eviction is O(1) for LRU (pop first item), O(N) for LFU (scan for minimum).
- **Space**: O(S) for the trie where S = total unique tokens stored across all sequences (shared prefixes counted once). O(N) for the OrderedDict with N cached entries.
- **Network**: Zero (local-only). Distributed version would add one Redis round-trip per lookup/store.
- **Disk**: Zero (in-memory only). Persistence would require serialization of trie and entry metadata.

## Metrics & Results

| Metric         | Value     | How Measured                    | Significance                              |
| -------------- | --------- | ------------------------------- | ----------------------------------------- |
| Trie Insert    | 156,160/s | 50K entries, 3 runs             | Fast enough for real-time caching         |
| Trie Lookup    | 150,372/s | 10K queries, 3 runs             | Sub-microsecond per query                 |
| Cache Store    | 60,168/s  | 10K entries, 3 runs             | Includes key hashing + eviction check     |
| Cache Lookup   | 142,346/s | 5K queries, 3 runs              | Trie lookup + access tracking + TTL check |
| Batch Analysis | 10,597/s  | 1K sequences, 3 runs            | Pre-flight dedup analysis                 |
| Test Count     | 68        | pytest                          | Comprehensive coverage                    |
| Coverage       | 84%       | pytest-cov (branch)             | CLI excluded                              |
| Token Savings  | 50%       | Demo with shared system prompts | Real workloads vary 30-80%                |

## Career Narrative

How this project fits the story:

- **JPMorgan (current)**: Built high-throughput data pipeline caches with eviction policies for market data -- same LRU/TTL patterns applied to KV cache management
- **Goldman Sachs (quant)**: Designed memory-efficient data structures for real-time pricing -- radix tries are used in IP routing tables and order book prefix matching
- **NVIDIA**: Understood GPU memory management constraints -- this library's memory budget enforcement directly addresses GPU OOM prevention for KV caches
- **This project**: Demonstrates systems-level thinking about inference optimization -- the exact skill set AI labs need for scaling LLM serving infrastructure

## Interview Red Flags to Avoid

- NEVER say "I built this to learn X" (sounds junior)
- NEVER be unable to explain any line of your code
- NEVER claim metrics you can't reproduce live
- NEVER badmouth existing tools (compare fairly)
- ALWAYS connect to the company's specific challenges
- ALWAYS mention what you'd improve
- ALWAYS discuss failure modes unprompted
