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
public class DefaultModelCache<T, TInput, TOutput> : IModelCache<T, TInput, TOutput>
{
    /// <summary>
    /// The internal dictionary that stores optimization step data, allowing concurrent access from multiple threads.
    /// </summary>
    /// <remarks>
    /// ConcurrentDictionary is used to ensure thread safety when multiple operations might access the cache simultaneously.
    /// </remarks>
    private readonly ConcurrentDictionary<string, OptimizationStepData<T, TInput, TOutput>> _cache = new();

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
        _cache.Clear();
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

        return _cache.TryGetValue(key, out var model) ? model : null;
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

        _cache[key] = stepData;
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
        Vector<T> parameters = solution.GetParameters();

        // Create stable descriptor of input data structure
        string inputDataDescriptor = DeterministicCacheKeyGenerator.CreateInputDataDescriptor<T, TInput, TOutput>(
            inputData.XTrain, inputData.YTrain,
            inputData.XValidation, inputData.YValidation,
            inputData.XTest, inputData.YTest);

        // Generate deterministic SHA-256 based key
        return DeterministicCacheKeyGenerator.GenerateKey<T>(parameters, inputDataDescriptor);
    }
}
