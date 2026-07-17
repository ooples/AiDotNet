using System.Collections.Generic;
using System.Text;
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
    private readonly PagedKVCache<T> _cache;
    private readonly int _capacity;
    private readonly object _lock = new();
    private readonly Dictionary<string, long> _registry = new();
    private readonly LinkedList<string> _lru = new();

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
            for (int len = promptTokens.Count - 1; len >= 1; len--)
            {
                string key = PrefixKey(promptTokens, len);
                if (_registry.TryGetValue(key, out long baseSeqId) &&
                    _cache.ForkSequence(baseSeqId, targetSequenceId))
                {
                    Touch(key);
                    return len;
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

        string key = PrefixKey(promptTokens, promptTokens.Count);
        lock (_lock)
        {
            if (_registry.ContainsKey(key))
            {
                Touch(key);
                return;
            }

            long baseSeqId = Interlocked.Decrement(ref _nextBaseId); // -1, -2, -3, ...
            if (!_cache.ForkSequence(sourceSequenceId, baseSeqId))
            {
                return; // best-effort: skip registration if the fork fails
            }

            _registry[key] = baseSeqId;
            _lru.AddLast(key);
            EvictIfNeeded();
        }
    }

    // Caller must hold _lock.
    private void Touch(string key)
    {
        _lru.Remove(key);
        _lru.AddLast(key);
    }

    // Caller must hold _lock.
    private void EvictIfNeeded()
    {
        while (_registry.Count > _capacity && _lru.First is not null)
        {
            string oldest = _lru.First.Value;
            _lru.RemoveFirst();
            if (_registry.TryGetValue(oldest, out long evictedBase))
            {
                _registry.Remove(oldest);
                _cache.FreeSequence(evictedBase);
            }
        }
    }

    private static string PrefixKey(IReadOnlyList<int> tokens, int length)
    {
        var sb = new StringBuilder(length * 4);
        for (int i = 0; i < length; i++)
        {
            sb.Append(tokens[i]);
            sb.Append(',');
        }
        return sb.ToString();
    }
}
