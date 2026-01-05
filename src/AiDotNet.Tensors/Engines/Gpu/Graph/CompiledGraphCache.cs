namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Graph cache for reusing compiled graphs.
/// </summary>
public sealed class CompiledGraphCache : IDisposable
{
    private readonly Dictionary<string, CachedGraph> _cache = new();
    private readonly int _maxCacheSize;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Gets the number of graphs currently cached.
    /// </summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _cache.Count;
            }
        }
    }

    /// <summary>
    /// Creates a new graph cache.
    /// </summary>
    /// <param name="maxSize">Maximum number of graphs to cache.</param>
    public CompiledGraphCache(int maxSize = 100)
    {
        _maxCacheSize = maxSize;
    }

    /// <summary>
    /// Gets or compiles a graph with the specified key.
    /// </summary>
    /// <param name="key">Cache key for the graph.</param>
    /// <param name="buildFunc">Function to build the graph if not cached.</param>
    /// <returns>The cached or newly compiled graph.</returns>
    public ExecutionGraph GetOrCompile(string key, Func<ExecutionGraph> buildFunc)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            if (_cache.TryGetValue(key, out var cached))
            {
                cached.HitCount++;
                cached.LastAccess = DateTime.UtcNow;
                return cached.Graph;
            }

            // Evict if at capacity
            if (_cache.Count >= _maxCacheSize)
            {
                EvictLeastUsed();
            }

            var graph = buildFunc();
            _cache[key] = new CachedGraph
            {
                Graph = graph,
                CreatedAt = DateTime.UtcNow,
                LastAccess = DateTime.UtcNow,
                HitCount = 0
            };

            return graph;
        }
    }

    /// <summary>
    /// Tries to get a cached graph.
    /// </summary>
    /// <param name="key">Cache key.</param>
    /// <param name="graph">The cached graph if found.</param>
    /// <returns>True if the graph was found in cache.</returns>
    public bool TryGet(string key, out ExecutionGraph? graph)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            if (_cache.TryGetValue(key, out var cached))
            {
                cached.HitCount++;
                cached.LastAccess = DateTime.UtcNow;
                graph = cached.Graph;
                return true;
            }

            graph = null;
            return false;
        }
    }

    /// <summary>
    /// Adds a graph to the cache.
    /// </summary>
    /// <param name="key">Cache key.</param>
    /// <param name="graph">The graph to cache.</param>
    public void Add(string key, ExecutionGraph graph)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            if (_cache.Count >= _maxCacheSize)
            {
                EvictLeastUsed();
            }

            _cache[key] = new CachedGraph
            {
                Graph = graph,
                CreatedAt = DateTime.UtcNow,
                LastAccess = DateTime.UtcNow,
                HitCount = 0
            };
        }
    }

    /// <summary>
    /// Removes a graph from the cache.
    /// </summary>
    /// <param name="key">Cache key.</param>
    /// <returns>True if a graph was removed.</returns>
    public bool Remove(string key)
    {
        lock (_lock)
        {
            if (_cache.TryGetValue(key, out var cached))
            {
                cached.Graph.Dispose();
                return _cache.Remove(key);
            }
            return false;
        }
    }

    /// <summary>
    /// Clears all cached graphs.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            foreach (var cached in _cache.Values)
            {
                cached.Graph.Dispose();
            }
            _cache.Clear();
        }
    }

    private void EvictLeastUsed()
    {
        // Find least recently used entry with lowest hit count
        var lru = _cache
            .OrderBy(kv => kv.Value.HitCount)
            .ThenBy(kv => kv.Value.LastAccess)
            .FirstOrDefault();

        if (lru.Key != null)
        {
            lru.Value.Graph.Dispose();
            _cache.Remove(lru.Key);
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CompiledGraphCache));
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Clear();
    }

    private sealed class CachedGraph
    {
        public ExecutionGraph Graph { get; init; } = null!;
        public DateTime CreatedAt { get; init; }
        public DateTime LastAccess { get; set; }
        public int HitCount { get; set; }
    }
}
