using System;
using System.Collections.Generic;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// Automatic cross-request prefix caching (as in vLLM's APC): remembers the KV-cache blocks of previously-seen
/// prompt prefixes so a later request that begins with the same tokens reuses that cached KV instead of
/// recomputing it. Only block-aligned prefixes are cached, so a shared block contains exactly the shared tokens.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> many requests share a common beginning — the same system prompt, the same few-shot
/// examples, the same document. Without caching, the model re-reads that shared beginning for every request. This
/// cache keeps the "memory" (KV) of shared beginnings around, so the next request that starts the same way skips
/// straight past it. It holds a reference to those blocks to keep them alive, and evicts the least-recently-used
/// entries when it is full.</para>
/// <para>The cache pins blocks by forking them into an internal cache-owned sequence in the
/// <see cref="BlockManager"/>; eviction frees that sequence, releasing the reference (blocks are reclaimed once
/// no live sequence uses them too).</para>
/// </remarks>
public sealed class PrefixCache
{
    private sealed class Entry
    {
        public string CacheSequenceId = string.Empty;
        public int[] PrefixTokens = Array.Empty<int>();
        public LinkedListNode<long> LruNode = null!;
    }

    private readonly BlockManager _blocks;
    private readonly int _blockSize;
    private readonly int _capacity;
    private readonly Dictionary<long, List<Entry>> _byKey = new(); // hash -> entries (collision chain)
    private readonly Dictionary<long, Entry> _byLruId = new();
    private readonly LinkedList<long> _lru = new(); // front = most recently used
    private long _counter;

    /// <summary>Creates a prefix cache over a block manager.</summary>
    /// <param name="blockManager">The block manager whose blocks are cached.</param>
    /// <param name="blockSize">KV block size (prefix lengths are multiples of this).</param>
    /// <param name="capacity">Maximum number of cached prefixes (LRU eviction beyond it).</param>
    public PrefixCache(BlockManager blockManager, int blockSize, int capacity = 128)
    {
        _blocks = blockManager ?? throw new ArgumentNullException(nameof(blockManager));
        if (blockSize < 1) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
        _blockSize = blockSize;
        _capacity = capacity;
    }

    /// <summary>The number of cached prefixes.</summary>
    public int Count => _byLruId.Count;

    /// <summary>Result of a cache lookup.</summary>
    public readonly struct Hit
    {
        /// <summary>Creates a hit.</summary>
        public Hit(string cacheSequenceId, int prefixLength) { CacheSequenceId = cacheSequenceId; PrefixLength = prefixLength; }
        /// <summary>The cache-owned sequence id whose blocks hold the shared prefix (fork from it).</summary>
        public string CacheSequenceId { get; }
        /// <summary>Number of shared prefix tokens (a positive multiple of the block size).</summary>
        public int PrefixLength { get; }
    }

    /// <summary>
    /// Finds the longest cached block-aligned prefix of <paramref name="promptTokens"/>. Returns null on a miss.
    /// A hit is moved to most-recently-used.
    /// </summary>
    public Hit? Lookup(IReadOnlyList<int> promptTokens)
    {
        if (promptTokens is null) throw new ArgumentNullException(nameof(promptTokens));
        int maxLen = promptTokens.Count / _blockSize * _blockSize;
        for (int len = maxLen; len >= _blockSize; len -= _blockSize)
        {
            long key = HashPrefix(promptTokens, len);
            if (!_byKey.TryGetValue(key, out var chain)) continue;
            foreach (var entry in chain)
            {
                if (PrefixMatches(entry.PrefixTokens, promptTokens, len))
                {
                    Touch(entry);
                    return new Hit(entry.CacheSequenceId, len);
                }
            }
        }
        return null;
    }

    /// <summary>
    /// Caches every block-aligned prefix of a sequence's prompt by pinning its blocks (forking into cache-owned
    /// sequences), so a later request that shares any block-aligned prefix reuses that KV. No-op below one block;
    /// already-cached prefixes are just refreshed.
    /// </summary>
    public void Register(IReadOnlyList<int> promptTokens, string ownerSequenceId)
    {
        if (promptTokens is null) throw new ArgumentNullException(nameof(promptTokens));
        int maxLen = promptTokens.Count / _blockSize * _blockSize;

        for (int len = _blockSize; len <= maxLen; len += _blockSize)
        {
            long key = HashPrefix(promptTokens, len);
            if (TryGetEntry(key, promptTokens, len, out var existing)) { Touch(existing); continue; }

            // Pin these prefix blocks under a cache-owned sequence so they survive the owner being freed.
            long id = ++_counter;
            string cacheSeqId = "prefixcache:" + id.ToString();
            _blocks.ForkPrefix(ownerSequenceId, cacheSeqId, len);

            var prefix = new int[len];
            for (int i = 0; i < len; i++) prefix[i] = promptTokens[i];
            var entry = new Entry { CacheSequenceId = cacheSeqId, PrefixTokens = prefix };
            entry.LruNode = _lru.AddFirst(id);
            _byLruId[id] = entry;
            if (!_byKey.TryGetValue(key, out var chain)) { chain = new List<Entry>(1); _byKey[key] = chain; }
            chain.Add(entry);

            if (_byLruId.Count > _capacity) EvictLeastRecentlyUsed();
        }
    }

    private bool TryGetEntry(long key, IReadOnlyList<int> tokens, int len, out Entry entry)
    {
        entry = null!;
        if (!_byKey.TryGetValue(key, out var chain)) return false;
        foreach (var e in chain)
            if (PrefixMatches(e.PrefixTokens, tokens, len)) { entry = e; return true; }
        return false;
    }

    /// <summary>Evicts the least-recently-used entry (releasing its pinned blocks). Returns false if empty.</summary>
    public bool TryEvictOne()
    {
        if (_lru.Last is null) return false;
        EvictLeastRecentlyUsed();
        return true;
    }

    /// <summary>Frees all cached prefixes (releasing their pinned blocks).</summary>
    public void Clear()
    {
        foreach (var entry in _byLruId.Values) _blocks.Free(entry.CacheSequenceId);
        _byLruId.Clear();
        _byKey.Clear();
        _lru.Clear();
    }

    private void Touch(Entry entry)
    {
        _lru.Remove(entry.LruNode);
        _lru.AddFirst(entry.LruNode);
    }

    private void EvictLeastRecentlyUsed()
    {
        var node = _lru.Last;
        if (node is null) return;
        long id = node.Value;
        _lru.RemoveLast();
        if (!_byLruId.Remove(id, out var entry)) return;

        _blocks.Free(entry.CacheSequenceId);
        long key = HashPrefix(entry.PrefixTokens, entry.PrefixTokens.Length);
        if (_byKey.TryGetValue(key, out var chain))
        {
            chain.Remove(entry);
            if (chain.Count == 0) _byKey.Remove(key);
        }
    }

    private static long HashPrefix(IReadOnlyList<int> tokens, int len)
    {
        // FNV-1a over the first `len` token ids.
        unchecked
        {
            long hash = 1469598103934665603L;
            for (int i = 0; i < len; i++)
            {
                hash ^= (uint)tokens[i];
                hash *= 1099511628211L;
            }
            return hash;
        }
    }

    private static bool PrefixMatches(int[] cached, IReadOnlyList<int> tokens, int len)
    {
        if (cached.Length != len) return false;
        for (int i = 0; i < len; i++) if (cached[i] != tokens[i]) return false;
        return true;
    }
}
