# X Thread: prompt-cache-engine

**Tweet 1:**
60-80% of LLM prompts share the same system prompt prefix.

That means 60-80% of your KV cache computation is redundant.

I built an engine-agnostic prefix cache that works with any inference backend.

Code: github.com/jrajath94/prompt-cache-engine

**Tweet 2:**
The problem: vLLM's prefix caching is great, but it's locked to vLLM.

SGLang's RadixAttention is great, but it's locked to SGLang.

If you're using TGI, PyTorch, or even API providers -- you get nothing.

**Tweet 3:**
How it works:

1. Radix trie stores token sequences, compressing shared prefixes
2. O(L) longest-prefix lookup (L = token length)
3. LRU/LFU eviction with TTL expiry
4. Batch analysis shows dedup potential before you commit

All in pure Python, zero inference engine dependencies.

**Tweet 4:**
The non-obvious insight: the cache manager should NOT manage KV tensors.

It manages keys, metadata, and eviction policy. Your inference engine manages the actual GPU memory.

Separation of concerns means it works everywhere.

**Tweet 5:**
Benchmarks (Apple M2, Python 3.9):

- Trie insert: 156,160 entries/s
- Trie lookup: 150,372 queries/s
- Cache store: 60,168 entries/s
- Cache lookup: 142,346 queries/s

Sub-microsecond lookups. Your bottleneck is the model, not the cache.

**Tweet 6:**
68 tests, 84% coverage. LRU + LFU + TTL eviction policies.

Star it if you're optimizing LLM inference costs.

github.com/jrajath94/prompt-cache-engine

#AI #MachineLearning #LLM #InferenceOptimization #OpenSource #BuildInPublic
