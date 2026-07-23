using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Inference.PagedAttention;

/// <summary>
/// RadixAttention-style prompt-prefix sharing over a shared <see cref="PagedKVCache{T}"/>. Maps a
/// prompt-prefix key to a base cache sequence whose KV holds exactly that prefix; new prompts fork the
/// longest registered strict prefix (copy-on-write) so the shared prefix KV is reused and only the
/// suffix allocates new blocks. LRU-capped; evicted bases are freed (existing forks keep shared blocks
/// alive via the paged cache's block ref-counting).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Many requests start with the same text (a shared system prompt, a common
/// few-shot preamble). Recomputing that shared start for every request wastes time and memory. This
/// cache remembers the key/value tensors ("KV") already computed for a shared beginning and lets a new
/// request <i>reuse</i> them instead of recomputing — it only has to process the part that is new.</para>
/// <para>
/// This is the single, shared prefix registry for one served model: both the ergonomic single-sequence
/// path and the continuous-batching engine use the SAME instance over the SAME paged cache, so a prefix
/// registered by one request is reusable by any later request regardless of which path served it.
/// </para>
/// <para><b>Sequence-id namespace.</b> Base sequence ids are minted from a strictly negative,
/// monotonically decreasing counter, disjoint from the positive ids the batcher assigns to live
/// sequences. This guarantees a base id can never collide with a live sequence id in the shared cache.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
internal sealed class RadixPrefixCache<T>
{
    // One trie node per distinct token position along a registered prefix path. A node carries a base
    // sequence id iff the path from the root to it is a registered prefix. This lets lookup walk the prompt
    // ONCE (O(n)) instead of reconstructing and hashing an O(len) key for every candidate length (O(n^2)).
    private sealed class RadixNode
    {
        internal readonly RadixNode? Parent;
        internal readonly int Token; // the token leading from Parent to this node (unused on the root)
        internal Dictionary<int, RadixNode>? Children;
        internal long? BaseSequenceId;
        internal LinkedListNode<RadixNode>? LruNode; // non-null iff this node is a registered prefix

        internal RadixNode(RadixNode? parent, int token)
        {
            Parent = parent;
            Token = token;
        }
    }

    private readonly PagedKVCache<T> _cache;
    private readonly int _capacity;
    private readonly object _lock = new();
    private readonly RadixNode _root = new(null, 0);
    // LRU over the registered-prefix nodes (each carries a BaseSequenceId); front = least-recently-used.
    // Count == number of registered prefixes, so capacity is enforced against _lru.Count.
    private readonly LinkedList<RadixNode> _lru = new();

    // Base sequence ids live in a strictly negative range, disjoint from the positive live-sequence ids
    // the batcher allocates — so a prefix base can never collide with an in-flight sequence.
    private long _nextBaseId = 0;

    /// <summary>Default number of distinct prompt prefixes retained before LRU eviction.</summary>
    public const int DefaultCapacity = 64;

    /// <summary>
    /// Creates a prefix cache over the given paged KV cache.
    /// </summary>
    /// <param name="cache">The shared paged KV cache backing all sequences of this model.</param>
    /// <param name="capacity">Maximum number of distinct prefixes to retain (LRU). Defaults to
    /// <see cref="DefaultCapacity"/>.</param>
    public RadixPrefixCache(PagedKVCache<T> cache, int capacity = DefaultCapacity)
    {
        Guard.NotNull(cache);
        _cache = cache;
        _capacity = capacity > 0 ? capacity : DefaultCapacity;
    }

    /// <summary>
    /// Forks the longest registered STRICT prefix of <paramref name="promptTokens"/> into
    /// <paramref name="targetSequenceId"/> (which the caller has already allocated), reusing that
    /// prefix's KV copy-on-write. Only a strict prefix is reused, so at least the final prompt token is
    /// still forwarded (producing the first next-token logits).
    /// </summary>
    /// <param name="promptTokens">The full prompt token ids.</param>
    /// <param name="targetSequenceId">The live sequence id to fork the prefix into.</param>
    /// <returns>The number of leading prompt tokens whose KV is now cached in
    /// <paramref name="targetSequenceId"/> (0 when no registered prefix matched).</returns>
    public int TryForkLongestPrefix(IReadOnlyList<int> promptTokens, long targetSequenceId)
    {
        Guard.NotNull(promptTokens);
        if (promptTokens.Count < 2)
        {
            return 0;
        }

        lock (_lock)
        {
            // Walk the prompt once, collecting the registered STRICT-prefix nodes (depth 1..count-1) along
            // the path. If a token has no matching child the path breaks — no longer prefix can match, since
            // any registered prefix shares this prompt's exact leading tokens.
            List<RadixNode>? candidates = null;
            var node = _root;
            int limit = promptTokens.Count - 1; // strict prefix: never reuse the full prompt
            for (int depth = 1; depth <= limit; depth++)
            {
                if (node.Children is null ||
                    !node.Children.TryGetValue(promptTokens[depth - 1], out var child))
                {
                    break;
                }
                node = child;
                if (node.BaseSequenceId.HasValue)
                {
                    (candidates ??= new List<RadixNode>()).Add(node);
                }
            }

            if (candidates is null)
            {
                return 0;
            }

            // Try the deepest registered prefix first; on a fork failure fall back to a shallower one
            // (preserving the previous longest-that-forks semantics). Depth of a node == its prefix length.
            for (int i = candidates.Count - 1; i >= 0; i--)
            {
                var cand = candidates[i];
                if (_cache.ForkSequence(cand.BaseSequenceId!.Value, targetSequenceId))
                {
                    Touch(cand);
                    return Depth(cand);
                }
            }
        }

        return 0;
    }

    /// <summary>
    /// Registers the full prompt as a reusable prefix. <paramref name="sourceSequenceId"/> must currently
    /// hold exactly the prompt's KV (prefill complete, no generated tokens yet); its KV is forked into a
    /// fresh negative base id stored under the prompt key so later prompts that extend it can fork.
    /// Best-effort: a failed fork or an already-registered key is a no-op.
    /// </summary>
    /// <param name="promptTokens">The full prompt token ids.</param>
    /// <param name="sourceSequenceId">A sequence whose KV holds exactly this prompt.</param>
    public void Register(IReadOnlyList<int> promptTokens, long sourceSequenceId)
    {
        Guard.NotNull(promptTokens);
        if (promptTokens.Count == 0)
        {
            return;
        }

        lock (_lock)
        {
            // Walk/create the trie path for the full prompt.
            var node = _root;
            for (int i = 0; i < promptTokens.Count; i++)
            {
                node.Children ??= new Dictionary<int, RadixNode>();
                if (!node.Children.TryGetValue(promptTokens[i], out var child))
                {
                    child = new RadixNode(node, promptTokens[i]);
                    node.Children[promptTokens[i]] = child;
                }
                node = child;
            }

            if (node.BaseSequenceId.HasValue)
            {
                Touch(node); // already registered — just refresh its LRU position
                return;
            }

            long baseSeqId = Interlocked.Decrement(ref _nextBaseId); // -1, -2, -3, ...
            if (!_cache.ForkSequence(sourceSequenceId, baseSeqId))
            {
                PruneIfEmpty(node); // best-effort: drop the dead path we may have just created
                return;
            }

            node.BaseSequenceId = baseSeqId;
            node.LruNode = _lru.AddLast(node);
            EvictIfNeeded();
        }
    }

    // Caller must hold _lock. Distance from the root == the node's registered-prefix length.
    private static int Depth(RadixNode node)
    {
        int depth = 0;
        for (var n = node; n.Parent is not null; n = n.Parent)
        {
            depth++;
        }
        return depth;
    }

    // Caller must hold _lock.
    private void Touch(RadixNode node)
    {
        if (node.LruNode is not null)
        {
            _lru.Remove(node.LruNode);
            node.LruNode = _lru.AddLast(node);
        }
    }

    // Caller must hold _lock.
    private void EvictIfNeeded()
    {
        while (_lru.Count > _capacity && _lru.First is not null)
        {
            var oldest = _lru.First.Value;
            _lru.RemoveFirst();
            oldest.LruNode = null;
            if (oldest.BaseSequenceId is { } evictedBase)
            {
                oldest.BaseSequenceId = null;
                _cache.FreeSequence(evictedBase);
            }
            PruneIfEmpty(oldest); // reclaim now-dead trie nodes so the trie can't grow unbounded
        }
    }

    // Caller must hold _lock. Removes childless, unregistered nodes walking up toward the root so an evicted
    // (or failed-to-register) leaf does not leave dangling internal nodes behind.
    private static void PruneIfEmpty(RadixNode node)
    {
        while (node.Parent is not null &&
               !node.BaseSequenceId.HasValue &&
               (node.Children is null || node.Children.Count == 0))
        {
            node.Parent.Children!.Remove(node.Token);
            node = node.Parent;
        }
    }
}
