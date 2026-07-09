global using System.Collections.Concurrent;
using System.Collections.Generic;
using AiDotNet.Enums;

using CacheEvictionPolicy = AiDotNet.Enums.CacheEvictionPolicy;
namespace AiDotNet.Caching;

/// <summary>
/// A thread-safe, string-keyed store bounded to a fixed capacity that evicts an entry per a
/// configurable <see cref="CacheEvictionPolicy"/> once full. Shared by the optimizer's gradient and
/// model-evaluation caches so the bounded-eviction logic lives in exactly one place instead of being
/// re-implemented per cache.
/// </summary>
/// <typeparam name="TValue">The cached value type (a reference type — the caches store class payloads).</typeparam>
/// <remarks>
/// Eviction victim selection:
/// <list type="bullet">
/// <item><description><b>FIFO</b> — the oldest <i>inserted</i> key (access does not change position).</description></item>
/// <item><description><b>LRU</b> — the least-recently <i>used</i> key (a get or an update refreshes recency).</description></item>
/// <item><description><b>LFU</b> — the least-frequently <i>used</i> key (fewest gets/updates; ties broken by age).</description></item>
/// </list>
/// All mutations run under a single lock so the capacity bound holds under concurrency; FIFO reads are
/// lock-free because they never mutate eviction metadata.
/// </remarks>
internal sealed class BoundedEvictingStore<TValue> where TValue : class
{
    private readonly ConcurrentDictionary<string, TValue> _values = new(StringComparer.Ordinal);
    // Insertion tick (FIFO) or last-access tick (LRU). Mutated only under _lock.
    private readonly Dictionary<string, long> _tick = new(StringComparer.Ordinal);
    // Access frequency (LFU). Mutated only under _lock.
    private readonly Dictionary<string, long> _frequency = new(StringComparer.Ordinal);
    private readonly object _lock = new();
    private long _clock;

    /// <summary>Maximum number of distinct keys retained before an eviction occurs.</summary>
    public int Capacity { get; }

    /// <summary>The policy that chooses the eviction victim once the store is over capacity.</summary>
    public CacheEvictionPolicy EvictionPolicy { get; }

    /// <summary>Current number of live entries.</summary>
    public int Count => _values.Count;

    public BoundedEvictingStore(int capacity, CacheEvictionPolicy evictionPolicy)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Cache capacity must be positive.");
        Capacity = capacity;
        EvictionPolicy = evictionPolicy;
    }

    /// <summary>Returns the value for <paramref name="key"/>, or <c>null</c> if absent.</summary>
    public TValue? Get(string key)
    {
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (!_values.TryGetValue(key, out var value))
            return null;

        // FIFO is access-order-agnostic, so its read stays lock-free. LRU/LFU must record the access.
        if (EvictionPolicy != CacheEvictionPolicy.FIFO)
        {
            lock (_lock)
            {
                if (_values.ContainsKey(key))
                    RecordUseUnlocked(key);
            }
        }
        return value;
    }

    /// <summary>Inserts or updates <paramref name="key"/>, evicting per policy when over capacity.</summary>
    public void Set(string key, TValue value)
    {
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));

        lock (_lock)
        {
            bool isNew = !_values.ContainsKey(key);
            _values[key] = value;

            if (isNew)
            {
                // A brand-new key: seed its metadata and evict back to capacity.
                _tick[key] = ++_clock;
                _frequency[key] = 1;
                while (_values.Count > Capacity)
                {
                    string victim = SelectVictimUnlocked();
                    _values.TryRemove(victim, out _);
                    _tick.Remove(victim);
                    _frequency.Remove(victim);
                }
            }
            else
            {
                // An in-place update counts as a use for LRU/LFU; FIFO keeps the original position.
                RecordUseUnlocked(key);
            }
        }
    }

    /// <summary>Removes every entry.</summary>
    public void Clear()
    {
        lock (_lock)
        {
            _values.Clear();
            _tick.Clear();
            _frequency.Clear();
        }
    }

    // Refresh the access metadata for a key already present. Caller holds _lock.
    private void RecordUseUnlocked(string key)
    {
        switch (EvictionPolicy)
        {
            case CacheEvictionPolicy.LRU:
                _tick[key] = ++_clock;
                break;
            case CacheEvictionPolicy.LFU:
                _frequency[key] = _frequency.TryGetValue(key, out var f) ? f + 1 : 1;
                break;
            // FIFO: insertion order is fixed; a use does not move the key.
        }
    }

    // Pick the eviction victim per policy. Caller holds _lock; _tick is never empty here (over capacity).
    private string SelectVictimUnlocked()
    {
        string? victim = null;
        long bestTick = long.MaxValue;
        long bestFreq = long.MaxValue;
        foreach (var entry in _tick)
        {
            long tick = entry.Value;
            long freq = _frequency.TryGetValue(entry.Key, out var f) ? f : 0;
            bool better = EvictionPolicy == CacheEvictionPolicy.LFU
                ? victim is null || freq < bestFreq || (freq == bestFreq && tick < bestTick)  // fewest uses, then oldest
                : victim is null || tick < bestTick;                                          // FIFO/LRU: smallest tick
            if (better)
            {
                victim = entry.Key;
                bestTick = tick;
                bestFreq = freq;
            }
        }
        return victim ?? throw new InvalidOperationException("Eviction requested on an empty cache.");
    }
}
