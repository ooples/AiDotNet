namespace AiDotNet.Diffusion.Acceleration;

/// <summary>
/// Pyramid Attention Broadcast (PAB) cache for accelerating video diffusion inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Real-Time Video Generation with Pyramid Attention Broadcast" (2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> PAB (Pyramid Attention Broadcast) Cache speeds up video generation by reusing attention computations across nearby diffusion timesteps. Since adjacent steps produce similar attention patterns, caching avoids redundant computation.</para>
/// <para>
/// PAB exploits the observation that attention outputs in video diffusion models change slowly
/// across denoising timesteps. Instead of recomputing attention at every step, PAB broadcasts
/// (reuses) cached attention outputs for a configurable number of steps:
/// - Spatial attention: changes most slowly, can be broadcast for many steps
/// - Temporal attention: changes moderately
/// - Cross-attention: changes most frequently, broadcast for fewer steps
/// This pyramid of broadcast intervals achieves significant speedup with minimal quality loss.
/// </para>
/// </remarks>
public class PABCache<T>
{
    private readonly int _spatialBroadcastInterval;
    private readonly int _temporalBroadcastInterval;
    private readonly int _crossBroadcastInterval;
    private readonly Dictionary<string, Tensor<T>> _spatialCache;
    private readonly Dictionary<string, Tensor<T>> _temporalCache;
    private readonly Dictionary<string, Tensor<T>> _crossCache;
    private int _currentStep;

    /// <summary>
    /// Gets the spatial attention broadcast interval.
    /// </summary>
    public int SpatialBroadcastInterval => _spatialBroadcastInterval;

    /// <summary>
    /// Gets the temporal attention broadcast interval.
    /// </summary>
    public int TemporalBroadcastInterval => _temporalBroadcastInterval;

    /// <summary>
    /// Gets the cross-attention broadcast interval.
    /// </summary>
    public int CrossBroadcastInterval => _crossBroadcastInterval;

    /// <summary>
    /// Gets the current denoising step.
    /// </summary>
    public int CurrentStep => _currentStep;

    /// <summary>
    /// Initializes a new PAB cache.
    /// </summary>
    /// <param name="spatialBroadcastInterval">Steps between spatial attention recomputation (default: 5).</param>
    /// <param name="temporalBroadcastInterval">Steps between temporal attention recomputation (default: 3).</param>
    /// <param name="crossBroadcastInterval">Steps between cross-attention recomputation (default: 1).</param>
    public PABCache(
        int spatialBroadcastInterval = 5,
        int temporalBroadcastInterval = 3,
        int crossBroadcastInterval = 1)
    {
        if (spatialBroadcastInterval <= 0)
            throw new ArgumentOutOfRangeException(nameof(spatialBroadcastInterval), "Broadcast interval must be positive.");
        if (temporalBroadcastInterval <= 0)
            throw new ArgumentOutOfRangeException(nameof(temporalBroadcastInterval), "Broadcast interval must be positive.");
        if (crossBroadcastInterval <= 0)
            throw new ArgumentOutOfRangeException(nameof(crossBroadcastInterval), "Broadcast interval must be positive.");

        _spatialBroadcastInterval = spatialBroadcastInterval;
        _temporalBroadcastInterval = temporalBroadcastInterval;
        _crossBroadcastInterval = crossBroadcastInterval;
        _spatialCache = new Dictionary<string, Tensor<T>>();
        _temporalCache = new Dictionary<string, Tensor<T>>();
        _crossCache = new Dictionary<string, Tensor<T>>();
        _currentStep = 0;
    }

    /// <summary>
    /// Checks if spatial attention should be recomputed at the current step.
    /// </summary>
    /// <param name="layerKey">Unique identifier for the layer.</param>
    /// <returns>True if recomputation is needed.</returns>
    public bool ShouldRecomputeSpatial(string layerKey)
    {
        Guard.NotNull(layerKey, nameof(layerKey));
        return _currentStep % _spatialBroadcastInterval == 0 || !_spatialCache.ContainsKey(layerKey);
    }

    /// <summary>
    /// Checks if temporal attention should be recomputed at the current step.
    /// </summary>
    /// <param name="layerKey">Unique identifier for the layer.</param>
    /// <returns>True if recomputation is needed.</returns>
    public bool ShouldRecomputeTemporal(string layerKey)
    {
        Guard.NotNull(layerKey, nameof(layerKey));
        return _currentStep % _temporalBroadcastInterval == 0 || !_temporalCache.ContainsKey(layerKey);
    }

    /// <summary>
    /// Checks if cross-attention should be recomputed at the current step.
    /// </summary>
    /// <param name="layerKey">Unique identifier for the layer.</param>
    /// <returns>True if recomputation is needed.</returns>
    public bool ShouldRecomputeCross(string layerKey)
    {
        Guard.NotNull(layerKey, nameof(layerKey));
        return _currentStep % _crossBroadcastInterval == 0 || !_crossCache.ContainsKey(layerKey);
    }

    /// <summary>
    /// Caches spatial attention output.
    /// </summary>
    public void CacheSpatial(string layerKey, Tensor<T> output)
    {
        Guard.NotNull(layerKey, nameof(layerKey));
        _spatialCache[layerKey] = output;
    }

    /// <summary>
    /// Caches temporal attention output.
    /// </summary>
    public void CacheTemporal(string layerKey, Tensor<T> output)
    {
        Guard.NotNull(layerKey, nameof(layerKey));
        _temporalCache[layerKey] = output;
    }

    /// <summary>
    /// Caches cross-attention output.
    /// </summary>
    public void CacheCross(string layerKey, Tensor<T> output)
    {
        Guard.NotNull(layerKey, nameof(layerKey));
        _crossCache[layerKey] = output;
    }

    /// <summary>
    /// Gets cached spatial attention output.
    /// </summary>
    public Tensor<T>? GetCachedSpatial(string layerKey)
    {
        return _spatialCache.TryGetValue(layerKey, out var cached) ? cached : null;
    }

    /// <summary>
    /// Gets cached temporal attention output.
    /// </summary>
    public Tensor<T>? GetCachedTemporal(string layerKey)
    {
        return _temporalCache.TryGetValue(layerKey, out var cached) ? cached : null;
    }

    /// <summary>
    /// Gets cached cross-attention output.
    /// </summary>
    public Tensor<T>? GetCachedCross(string layerKey)
    {
        return _crossCache.TryGetValue(layerKey, out var cached) ? cached : null;
    }

    /// <summary>
    /// Advances to the next denoising step.
    /// </summary>
    public void Step()
    {
        _currentStep++;
    }

    /// <summary>
    /// Resets the cache for a new generation.
    /// </summary>
    public void Reset()
    {
        _currentStep = 0;
        _spatialCache.Clear();
        _temporalCache.Clear();
        _crossCache.Clear();
    }
}
