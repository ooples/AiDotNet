namespace AiDotNet.Data.Loaders;

/// <summary>
/// Wraps data loading with an in-memory cache to avoid redundant I/O.
/// </summary>
/// <remarks>
/// <para>
/// Caches loaded batches by their key (typically batch index) to avoid reloading
/// data that has already been seen. Useful when iterating over the same data
/// multiple epochs or when random access patterns cause repeated reads.
/// </para>
/// </remarks>
/// <typeparam name="TKey">The cache key type (typically int for batch index).</typeparam>
/// <typeparam name="TValue">The cached value type (typically a batch tensor).</typeparam>
public class CachingDataLoader<TKey, TValue> where TKey : notnull
{
    private readonly CachingDataLoaderOptions _options;
    private readonly Dictionary<TKey, CacheEntry> _cache;
    private readonly LinkedList<TKey> _accessOrder;
    private readonly object _lock = new();

    private sealed class CacheEntry
    {
        public TValue Value { get; set; } = default!;
        public int AccessCount { get; set; }
        public LinkedListNode<TKey>? OrderNode { get; set; }
    }

    /// <summary>
    /// Gets the number of items currently in the cache.
    /// </summary>
    public int Count
    {
        get { lock (_lock) return _cache.Count; }
    }

    /// <summary>
    /// Gets the cache hit ratio (hits / total requests).
    /// </summary>
    public double HitRatio
    {
        get
        {
            long total = Interlocked.Read(ref _totalRequests);
            long hits = Interlocked.Read(ref _hits);
            return total > 0 ? (double)hits / total : 0;
        }
    }

    private long _hits;
    private long _totalRequests;

    /// <summary>
    /// Creates a new caching data loader.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public CachingDataLoader(CachingDataLoaderOptions? options = null)
    {
        _options = options ?? new CachingDataLoaderOptions();
        _options.Validate();
        _cache = new Dictionary<TKey, CacheEntry>();
        _accessOrder = new LinkedList<TKey>();
    }

    /// <summary>
    /// Gets a value from the cache, or loads it using the provided factory.
    /// </summary>
    /// <param name="key">The cache key.</param>
    /// <param name="factory">Function to load the value if not cached.</param>
    /// <returns>The cached or freshly loaded value.</returns>
    public TValue GetOrLoad(TKey key, Func<TKey, TValue> factory)
    {
        if (factory == null) throw new ArgumentNullException(nameof(factory));

        lock (_lock)
        {
            Interlocked.Increment(ref _totalRequests);

            if (_cache.TryGetValue(key, out var entry))
            {
                Interlocked.Increment(ref _hits);
                entry.AccessCount++;

                // Move to front for LRU (but not FIFO - FIFO preserves insertion order)
                if (_options.EvictionPolicy != MemoryCacheEvictionPolicy.FIFO && entry.OrderNode != null)
                {
                    _accessOrder.Remove(entry.OrderNode);
                    _accessOrder.AddFirst(entry.OrderNode);
                }

                return entry.Value;
            }
        }

        // Load outside of lock
        TValue value = factory(key);

        lock (_lock)
        {
            // Double-check: another thread may have loaded same key
            if (_cache.TryGetValue(key, out var existing))
                return existing.Value;

            // Evict if needed (bound iterations to prevent infinite loop if Evict makes no progress)
            int evictAttempts = 0;
            int maxEvictAttempts = _cache.Count + 1;
            while (_cache.Count >= _options.MaxCacheSize && _cache.Count > 0 && evictAttempts < maxEvictAttempts)
            {
                int countBefore = _cache.Count;
                Evict();
                evictAttempts++;
                if (_cache.Count >= countBefore) break; // No progress — avoid infinite loop
            }

            var node = _accessOrder.AddFirst(key);
            _cache[key] = new CacheEntry
            {
                Value = value,
                AccessCount = 1,
                OrderNode = node
            };
        }

        return value;
    }

    /// <summary>
    /// Checks if a key is in the cache.
    /// </summary>
    /// <param name="key">The key to check.</param>
    /// <returns>True if the key is cached.</returns>
    public bool Contains(TKey key)
    {
        lock (_lock) return _cache.ContainsKey(key);
    }

    /// <summary>
    /// Clears all cached items and resets statistics.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _cache.Clear();
            _accessOrder.Clear();
            _hits = 0;
            _totalRequests = 0;
        }
    }

    /// <summary>
    /// Invalidates a specific cache entry.
    /// </summary>
    /// <param name="key">The key to invalidate.</param>
    /// <returns>True if the entry was found and removed.</returns>
    public bool Invalidate(TKey key)
    {
        lock (_lock)
        {
            if (_cache.TryGetValue(key, out var entry))
            {
                if (entry.OrderNode != null)
                    _accessOrder.Remove(entry.OrderNode);
                _cache.Remove(key);
                return true;
            }
            return false;
        }
    }

    private void Evict()
    {
        // Must be called under lock
        TKey? evictKey;

        switch (_options.EvictionPolicy)
        {
            case MemoryCacheEvictionPolicy.LRU:
                // Evict least recently used (tail of access order)
                var lruNode = _accessOrder.Last;
                if (lruNode == null) return;
                evictKey = lruNode.Value;
                _accessOrder.RemoveLast();
                break;

            case MemoryCacheEvictionPolicy.FIFO:
                // FIFO evicts the oldest inserted item (tail of list).
                // Unlike LRU, FIFO does not move items to front on access.
                var fifoNode = _accessOrder.Last;
                if (fifoNode == null) return;
                evictKey = fifoNode.Value;
                _accessOrder.RemoveLast();
                break;

            case MemoryCacheEvictionPolicy.LFU:
                // Find least frequently used
                int minAccess = int.MaxValue;
                evictKey = default;
                foreach (var kvp in _cache)
                {
                    if (kvp.Value.AccessCount < minAccess)
                    {
                        minAccess = kvp.Value.AccessCount;
                        evictKey = kvp.Key;
                    }
                }
                if (evictKey == null) return;
                if (_cache.TryGetValue(evictKey, out var lfuEntry) && lfuEntry.OrderNode != null)
                    _accessOrder.Remove(lfuEntry.OrderNode);
                break;

            default:
                return;
        }

        if (evictKey != null)
            _cache.Remove(evictKey);
    }
}
