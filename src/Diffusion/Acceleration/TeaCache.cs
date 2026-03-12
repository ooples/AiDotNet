namespace AiDotNet.Diffusion.Acceleration;

/// <summary>
/// Timestep Embedding Aware Cache (TeaCache) for accelerating video diffusion inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "TeaCache: Timestep-Aware KV-Cache for Efficient Video Diffusion" (2025)</item>
/// </list></para>
/// <para><b>For Beginners:</b> TeaCache (Timestep Embedding Aware Cache) accelerates DiT-based video generation by caching and reusing intermediate computations when timestep embeddings are similar. It provides significant speedup with minimal quality loss.</para>
/// <para>
/// TeaCache accelerates video diffusion inference by caching and reusing key-value pairs
/// across denoising timesteps. The key insight is that adjacent timesteps have very similar
/// attention patterns, so KV pairs can be reused with minimal quality degradation.
/// </para>
/// <para>
/// The cache uses a similarity threshold to determine when to recompute:
/// - If the timestep embedding change is small, reuse cached KV pairs
/// - If the change exceeds the threshold, recompute attention
/// - Achieves 2-3x speedup on typical 50-step diffusion sampling
/// </para>
/// </remarks>
public class TeaCache<T>
{
    private readonly double _reuseThreshold;
    private readonly int _maxCacheSize;
    private readonly Dictionary<string, Tensor<T>> _kvCache;
    private readonly Dictionary<string, double> _lastTimestepEmbedding;
    private readonly List<string> _insertionOrder;
    private int _cacheHits;
    private int _cacheMisses;

    /// <summary>
    /// Gets the reuse threshold for timestep embedding similarity.
    /// </summary>
    public double ReuseThreshold => _reuseThreshold;

    /// <summary>
    /// Gets the maximum cache size.
    /// </summary>
    public int MaxCacheSize => _maxCacheSize;

    /// <summary>
    /// Gets the number of cache hits.
    /// </summary>
    public int CacheHits => _cacheHits;

    /// <summary>
    /// Gets the number of cache misses.
    /// </summary>
    public int CacheMisses => _cacheMisses;

    /// <summary>
    /// Gets the cache hit rate.
    /// </summary>
    public double HitRate => _cacheHits + _cacheMisses > 0
        ? (double)_cacheHits / (_cacheHits + _cacheMisses)
        : 0.0;

    /// <summary>
    /// Initializes a new TeaCache.
    /// </summary>
    /// <param name="reuseThreshold">Similarity threshold for reusing cached values (0.0-1.0).</param>
    /// <param name="maxCacheSize">Maximum number of cached entries.</param>
    public TeaCache(
        double reuseThreshold = 0.05,
        int maxCacheSize = 256)
    {
        if (reuseThreshold < 0.0 || reuseThreshold > 1.0)
            throw new ArgumentOutOfRangeException(nameof(reuseThreshold), "Reuse threshold must be between 0.0 and 1.0.");
        if (maxCacheSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxCacheSize), "Max cache size must be positive.");

        _reuseThreshold = reuseThreshold;
        _maxCacheSize = maxCacheSize;
        _kvCache = new Dictionary<string, Tensor<T>>();
        _lastTimestepEmbedding = new Dictionary<string, double>();
        _insertionOrder = new List<string>();
        _cacheHits = 0;
        _cacheMisses = 0;
    }

    /// <summary>
    /// Checks if cached KV pairs can be reused for the given layer and timestep.
    /// </summary>
    /// <param name="layerKey">Unique identifier for the attention layer.</param>
    /// <param name="timestepEmbedding">Current timestep embedding magnitude.</param>
    /// <returns>True if cached values should be reused.</returns>
    public bool ShouldReuse(string layerKey, double timestepEmbedding)
    {
        if (!_kvCache.ContainsKey(layerKey) || !_lastTimestepEmbedding.TryGetValue(layerKey, out double lastEmbedding))
        {
            _cacheMisses++;
            return false;
        }

        double diff = Math.Abs(timestepEmbedding - lastEmbedding);
        if (diff < _reuseThreshold)
        {
            _cacheHits++;
            return true;
        }

        _cacheMisses++;
        return false;
    }

    /// <summary>
    /// Stores KV pairs in the cache.
    /// </summary>
    /// <param name="layerKey">Unique identifier for the attention layer.</param>
    /// <param name="kvPairs">The key-value tensor to cache.</param>
    /// <param name="timestepEmbedding">The timestep embedding magnitude when this was computed.</param>
    public void Store(string layerKey, Tensor<T> kvPairs, double timestepEmbedding)
    {
        // Evict oldest entry if cache is full (use explicit insertion order for net471 compatibility)
        if (_kvCache.Count >= _maxCacheSize && !_kvCache.ContainsKey(layerKey))
        {
            var oldestKey = _insertionOrder[0];
            _insertionOrder.RemoveAt(0);
            _kvCache.Remove(oldestKey);
            _lastTimestepEmbedding.Remove(oldestKey);
        }

        if (!_kvCache.ContainsKey(layerKey))
            _insertionOrder.Add(layerKey);

        _kvCache[layerKey] = kvPairs;
        _lastTimestepEmbedding[layerKey] = timestepEmbedding;
    }

    /// <summary>
    /// Retrieves cached KV pairs.
    /// </summary>
    /// <param name="layerKey">Unique identifier for the attention layer.</param>
    /// <returns>Cached KV tensor, or null if not found.</returns>
    public Tensor<T>? Retrieve(string layerKey)
    {
        return _kvCache.TryGetValue(layerKey, out var cached) ? cached : null;
    }

    /// <summary>
    /// Resets the cache and statistics.
    /// </summary>
    public void Reset()
    {
        _kvCache.Clear();
        _lastTimestepEmbedding.Clear();
        _insertionOrder.Clear();
        _cacheHits = 0;
        _cacheMisses = 0;
    }
}
