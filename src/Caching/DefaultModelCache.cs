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
    /// <returns>The cached optimization step data if found; otherwise, a new empty instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks up previously saved information about a specific training step.
    /// 
    /// The "key" is like a label that identifies which piece of information you want to retrieve.
    /// For example, the key might represent a specific parameter in your model or a particular
    /// point in the training process.
    /// 
    /// If the information exists in the cache, the method returns it. If not, it returns a new empty
    /// container that can be filled with fresh information.
    /// </para>
    /// </remarks>
    public OptimizationStepData<T, TInput, TOutput>? GetCachedStepData(string key)
    {
        return _cache.TryGetValue(key, out var model) ? model : new();
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
        _cache[key] = stepData;
    }

    /// <summary>
    /// Generates a unique cache key based on the model and input data.
    /// </summary>
    /// <param name="solution">The model configuration to generate a key for.</param>
    /// <param name="inputData">The input data to include in the key generation.</param>
    /// <returns>A unique string that identifies this specific model and data combination.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a unique identifier (like a fingerprint) for a specific
    /// combination of model settings and data. This "fingerprint" is used to check if we've already
    /// calculated results for this exact combination before.
    ///
    /// The key is generated from:
    /// - The model's parameters (if the model supports parameter access)
    /// - The model's hash code (a unique identifier for the model instance)
    /// - The input data's hash code
    ///
    /// If the same model and data are used again, the same key will be generated, allowing the system
    /// to retrieve previous results instead of recalculating.
    /// </para>
    /// </remarks>
    public string GenerateCacheKey(IFullModel<T, TInput, TOutput> solution, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Create a hash-based key from model parameters and input data
        var keyBuilder = new System.Text.StringBuilder();

        // Add model identifier
        keyBuilder.Append("Model:");
        keyBuilder.Append(solution.GetHashCode());
        keyBuilder.Append("|");

        // Add parameter information if available
        if (solution is IParameterizable<T, TInput, TOutput> parameterizable)
        {
            keyBuilder.Append("Params:");
            var parameters = parameterizable.GetParameters();
            if (parameters != null && parameters.Count > 0)
            {
                keyBuilder.Append(parameters.GetHashCode());
            }
            keyBuilder.Append("|");
        }

        // Add input data identifier
        keyBuilder.Append("Data:");
        keyBuilder.Append(inputData.GetHashCode());

        return keyBuilder.ToString();
    }
}