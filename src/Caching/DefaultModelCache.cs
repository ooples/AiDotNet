using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using CacheEvictionPolicy = AiDotNet.Enums.CacheEvictionPolicy;
namespace AiDotNet.Caching;

/// <summary>
/// Provides a default implementation of model caching for optimization step data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A model cache is like a storage box that keeps track of the progress made during 
/// machine learning model training.
/// 
/// When training a machine learning model, the system makes many small adjustments (optimization steps) 
/// to improve the model's accuracy. Each step produces important information about the model's current state.
/// 
/// This cache stores that information for each step, allowing the training process to:
/// - Resume training from where it left off if interrupted
/// - Avoid repeating calculations that were already done
/// - Keep track of how the model is improving over time
/// 
/// Think of it like saving your progress in a video game, so you don't have to start from the beginning 
/// if you need to take a break.
/// </para>
/// </remarks>
[InfraType(InfraType.Cache)]
public class DefaultModelCache<T, TInput, TOutput> : IModelCache<T, TInput, TOutput>
{
    /// <summary>
    /// Default maximum number of distinct step-data entries retained before eviction.
    /// </summary>
    public const int DefaultCapacity = 8;

    /// <summary>Recommended default eviction policy: oldest-inserted first.</summary>
    public const CacheEvictionPolicy DefaultEvictionPolicy = CacheEvictionPolicy.FIFO;

    /// <summary>Bounded, policy-evicting backing store (capacity + FIFO/LRU/LFU live here).</summary>
    private readonly BoundedEvictingStore<OptimizationStepData<T, TInput, TOutput>> _store;

    /// <summary>When false the cache is a pass-through: every lookup misses and stores are dropped.</summary>
    private readonly bool _enabled;

    /// <summary>Whether the cache actually stores entries (false = disabled pass-through).</summary>
    public bool Enabled => _enabled;

    /// <summary>Maximum number of distinct keys retained before an entry is evicted.</summary>
    public int Capacity => _store.Capacity;

    /// <summary>The policy used to choose the eviction victim once the cache is full.</summary>
    public CacheEvictionPolicy EvictionPolicy => _store.EvictionPolicy;

    /// <summary>
    /// Initializes a new cache with the recommended defaults: <see cref="DefaultCapacity"/> entries,
    /// <see cref="DefaultEvictionPolicy"/> eviction.
    /// </summary>
    public DefaultModelCache() : this(DefaultCapacity, DefaultEvictionPolicy)
    {
    }

    /// <summary>
    /// Initializes a new cache that retains at most <paramref name="capacity"/> distinct entries,
    /// evicting with the recommended <see cref="DefaultEvictionPolicy"/>.
    /// </summary>
    /// <param name="capacity">
    /// Maximum number of step-data entries to keep. This bounds memory: an optimizer's per-epoch
    /// evaluation writes a fresh entry every epoch (the parameter-content key changes as the model
    /// trains), so without a bound the cache — which retains a deep-copied model and O(N) predictions
    /// per entry — would grow without limit across a long training run. Values &lt;= 0 fall back to
    /// <see cref="DefaultCapacity"/>.
    /// </param>
    public DefaultModelCache(int capacity) : this(capacity, DefaultEvictionPolicy)
    {
    }

    /// <summary>
    /// Initializes a new cache that retains at most <paramref name="capacity"/> distinct entries,
    /// evicting per <paramref name="evictionPolicy"/>.
    /// </summary>
    /// <param name="capacity">Maximum entries to keep; values &lt;= 0 fall back to <see cref="DefaultCapacity"/>.</param>
    /// <param name="evictionPolicy">Which entry to evict once the cache is full (FIFO / LRU / LFU).</param>
    /// <param name="enabled">When false the cache is a pass-through that never stores (always recomputes).</param>
    public DefaultModelCache(int capacity, CacheEvictionPolicy evictionPolicy, bool enabled = true)
    {
        _store = new BoundedEvictingStore<OptimizationStepData<T, TInput, TOutput>>(
            capacity > 0 ? capacity : DefaultCapacity, evictionPolicy);
        _enabled = enabled;
    }

    /// <summary>
    /// Removes all cached optimization step data from the cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this method when you want to start fresh, such as when beginning a new training session
    /// with different parameters or a different dataset.
    /// </para>
    /// </remarks>
    public void ClearCache()
    {
        _store.Clear();
    }

    /// <summary>
    /// Retrieves cached optimization step data using the specified key.
    /// </summary>
    /// <param name="key">The unique identifier for the optimization step data.</param>
    /// <returns>The cached optimization step data if found; otherwise, null.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks up previously saved information about a specific training step.
    /// 
    /// The "key" is like a label that identifies which piece of information you want to retrieve.
    /// For example, the key might represent a specific parameter in your model or a particular
    /// point in the training process.
    /// 
    /// If the information exists in the cache, the method returns it. If not, it returns null
    /// indicating the data needs to be computed.
    /// </para>
    /// </remarks>
    public OptimizationStepData<T, TInput, TOutput>? GetCachedStepData(string key)
    {
        if (key == null) throw new ArgumentNullException(nameof(key), "Cache key cannot be null.");
        return _enabled ? _store.Get(key) : null;
    }

    /// <summary>
    /// Stores optimization step data in the cache with the specified key.
    /// </summary>
    /// <param name="key">The unique identifier for the optimization step data.</param>
    /// <param name="stepData">The optimization step data to cache.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method saves information about a training step so it can be used later.
    ///
    /// During model training, each step produces valuable information about how the model is changing
    /// and improving. This method stores that information with a unique label (the key) so you can
    /// retrieve it later.
    ///
    /// If data with the same key already exists in the cache, it will be replaced with the new data.
    /// This is useful for updating the cache with the latest information as training progresses.
    /// </para>
    /// </remarks>
    public void CacheStepData(string key, OptimizationStepData<T, TInput, TOutput> stepData)
    {
        if (key == null) throw new ArgumentNullException(nameof(key), "Cache key cannot be null.");
        if (stepData == null) throw new ArgumentNullException(nameof(stepData), "Step data cannot be null.");
        if (_enabled) _store.Set(key, stepData);
    }

    /// <summary>
    /// Generates a deterministic cache key based on the solution model and input data using SHA-256 hashing.
    /// </summary>
    /// <param name="solution">The model solution to generate a key for.</param>
    /// <param name="inputData">The input data to include in the key generation.</param>
    /// <returns>A deterministic hex-encoded SHA-256 hash string for caching.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a deterministic identifier based on the model and its inputs.
    /// </para>
    /// <para>
    /// Think of it like creating a fingerprint for a specific combination of model parameters and input data
    /// that stays the same forever, even if you restart the program. The same combination will always produce
    /// the same key, which allows the system to:
    /// - Save results with this key
    /// - Look up previously saved results using this key
    /// - Avoid recalculating results that have already been computed
    /// - Keep caches valid across application restarts
    /// </para>
    /// <para>
    /// The key is generated using SHA-256 cryptographic hashing for determinism:
    /// 1. Model parameters are serialized in a stable format with culture-invariant number formatting
    /// 2. Input data structure (shapes/dimensions) is described in a stable string format
    /// 3. SHA-256 hash is computed over the UTF-8 bytes of the serialized data
    /// 4. The hash is returned as a lowercase hexadecimal string
    /// </para>
    /// <para>
    /// This ensures that different combinations get different keys, while identical combinations always get
    /// the same key, even across process restarts.
    /// </para>
    /// </remarks>
    public string GenerateCacheKey(IFullModel<T, TInput, TOutput> solution, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (solution == null) throw new ArgumentNullException(nameof(solution));
        if (inputData == null) throw new ArgumentNullException(nameof(inputData));

        // Get solution parameters
        Vector<T> parameters = InterfaceGuard.Parameterizable(solution).GetParameters();

        // Create stable descriptor of input data structure
        string inputDataDescriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<T, TInput, TOutput>(
            inputData.XTrain, inputData.YTrain,
            inputData.XValidation, inputData.YValidation,
            inputData.XTest, inputData.YTest);

        // Generate deterministic SHA-256 based key
        return DeterministicCacheKeyGenerator.GenerateKey<T>(parameters, inputDataDescriptor);
    }
}
