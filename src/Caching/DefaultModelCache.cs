namespace AiDotNet.Caching;

/// <summary>
/// Provides a default implementation of model caching for optimization step data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// For Beginners: A model cache is like a storage box that keeps track of the progress made during 
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
public class DefaultModelCache<T> : IModelCache<T>
{
    /// <summary>
    /// The internal dictionary that stores optimization step data, allowing concurrent access from multiple threads.
    /// </summary>
    /// <remarks>
    /// ConcurrentDictionary is used to ensure thread safety when multiple operations might access the cache simultaneously.
    /// </remarks>
    private readonly ConcurrentDictionary<string, OptimizationStepData<T>> _cache = new();

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
    /// For Beginners: This method looks up previously saved information about a specific training step.
    /// 
    /// The "key" is like a label that identifies which piece of information you want to retrieve.
    /// For example, the key might represent a specific parameter in your model or a particular
    /// point in the training process.
    /// 
    /// If the information exists in the cache, the method returns it. If not, it returns a new empty
    /// container that can be filled with fresh information.
    /// </para>
    /// </remarks>
    public OptimizationStepData<T>? GetCachedStepData(string key)
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
    /// For Beginners: This method saves information about a training step so it can be used later.
    /// 
    /// During model training, each step produces valuable information about how the model is changing
    /// and improving. This method stores that information with a unique label (the key) so you can
    /// retrieve it later.
    /// 
    /// If data with the same key already exists in the cache, it will be replaced with the new data.
    /// This is useful for updating the cache with the latest information as training progresses.
    /// </para>
    /// </remarks>
    public void CacheStepData(string key, OptimizationStepData<T> stepData)
    {
        _cache[key] = stepData;
    }
}