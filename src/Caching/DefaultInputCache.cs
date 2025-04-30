namespace AiDotNet.Caching;

/// <summary>
/// Provides a cache for optimization input data that can be accessed globally throughout the application.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This cache stores input data that's used during model optimization.
/// 
/// Since most models typically use the same input data (from the PredictionModelBuilder),
/// this cache primarily stores that default data. However, it also lets you store and
/// retrieve alternative datasets for experiments with different data splits,
/// normalization methods, or other preprocessing variations.
/// 
/// Think of it like having a main recipe that most chefs follow, but also allowing
/// individual chefs to create and use their own recipe variations when they want to.
/// </remarks>
public static class DefaultInputCache
{
    /// <summary>
    /// The internal cache that stores named input data sets.
    /// </summary>
    private static readonly ConcurrentDictionary<string, object> _cache = new();

    /// <summary>
    /// The name of the default entry used for the main optimization input data.
    /// </summary>
    public const string DefaultCacheKey = "DefaultOptimizationInputData";

    /// <summary>
    /// Stores the default input data that most models will use.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TInput">The type of input data.</typeparam>
    /// <typeparam name="TOutput">The type of output data.</typeparam>
    /// <param name="inputData">The input data to cache as the default.</param>
    /// <exception cref="ArgumentNullException">Thrown if inputData is null.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method stores the main input data that will be used by most models.
    /// Typically, this is called from the PredictionModelBuilder when setting up the optimization process.
    /// </remarks>
    public static void CacheDefaultInputData<T, TInput, TOutput>(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        _cache[DefaultCacheKey] = inputData;
    }

    /// <summary>
    /// Stores custom input data with a specified name for specialized use cases.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TInput">The type of input data.</typeparam>
    /// <typeparam name="TOutput">The type of output data.</typeparam>
    /// <param name="name">A unique name to identify this input data set.</param>
    /// <param name="inputData">The input data to cache.</param>
    /// <exception cref="ArgumentNullException">Thrown if name or inputData is null.</exception>
    /// <exception cref="ArgumentException">Thrown if name is empty or just whitespace.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method lets you store alternative input data sets with custom names.
    /// This is useful when you want to experiment with different preprocessing techniques,
    /// data splits, or normalization methods without affecting the default data that most models use.
    /// </remarks>
    public static void CacheNamedInputData<T, TInput, TOutput>(string name, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Name cannot be empty or whitespace.", nameof(name));
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        _cache[name] = inputData;
    }

    /// <summary>
    /// Retrieves the default cached input data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TInput">The type of input data.</typeparam>
    /// <typeparam name="TOutput">The type of output data.</typeparam>
    /// <returns>The default input data.</returns>
    /// <exception cref="InvalidOperationException">Thrown if no default input data has been cached.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method retrieves the main input data that most models use.
    /// It's the data that was stored using CacheDefaultInputData.
    /// </remarks>
    public static OptimizationInputData<T, TInput, TOutput> GetDefaultInputData<T, TInput, TOutput>()
    {
        if (_cache.TryGetValue(DefaultCacheKey, out var cachedData) &&
            cachedData is OptimizationInputData<T, TInput, TOutput> typedData)
        {
            return typedData;
        }

        throw new InvalidOperationException(
            "No default input data has been cached. " +
            "Call CacheDefaultInputData first to initialize the default cache.");
    }

    /// <summary>
    /// Retrieves named input data from the cache.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TInput">The type of input data.</typeparam>
    /// <typeparam name="TOutput">The type of output data.</typeparam>
    /// <param name="name">The name of the cached input data to retrieve.</param>
    /// <returns>The cached input data with the specified name.</returns>
    /// <exception cref="ArgumentNullException">Thrown if name is null.</exception>
    /// <exception cref="ArgumentException">Thrown if name is empty or just whitespace.</exception>
    /// <exception cref="InvalidOperationException">Thrown if no input data with the specified name exists in the cache.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method retrieves custom input data that you stored with a specific name.
    /// If you've created different versions of your input data for experimentation,
    /// you can access them using the same name you gave them when caching.
    /// </remarks>
    public static OptimizationInputData<T, TInput, TOutput> GetNamedInputData<T, TInput, TOutput>(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Name cannot be empty or whitespace.", nameof(name));

        if (_cache.TryGetValue(name, out var cachedData) &&
            cachedData is OptimizationInputData<T, TInput, TOutput> typedData)
        {
            return typedData;
        }

        throw new InvalidOperationException(
            $"No input data named '{name}' has been cached. " +
            $"Call CacheNamedInputData first to cache data with this name.");
    }

    /// <summary>
    /// Attempts to retrieve the default input data from the cache.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TInput">The type of input data.</typeparam>
    /// <typeparam name="TOutput">The type of output data.</typeparam>
    /// <param name="inputData">When this method returns, contains the default input data
    /// if found; otherwise, the default value.</param>
    /// <returns>true if default input data was found; otherwise, false.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is a safer version of GetDefaultInputData that won't throw
    /// an error if no default data has been cached. It's useful when you want to check
    /// if default data exists before deciding whether to use it or create new data.
    /// </remarks>
    public static bool TryGetDefaultInputData<T, TInput, TOutput>(out OptimizationInputData<T, TInput, TOutput>? inputData)
    {
        inputData = null;

        if (_cache.TryGetValue(DefaultCacheKey, out var cachedData) &&
            cachedData is OptimizationInputData<T, TInput, TOutput> typedData)
        {
            inputData = typedData;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Attempts to retrieve named input data from the cache.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TInput">The type of input data.</typeparam>
    /// <typeparam name="TOutput">The type of output data.</typeparam>
    /// <param name="name">The name of the cached input data to retrieve.</param>
    /// <param name="inputData">When this method returns, contains the input data with the specified name
    /// if found; otherwise, the default value.</param>
    /// <returns>true if input data with the specified name was found; otherwise, false.</returns>
    /// <exception cref="ArgumentNullException">Thrown if name is null.</exception>
    /// <exception cref="ArgumentException">Thrown if name is empty or just whitespace.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This is a safer version of GetNamedInputData that won't throw
    /// an error if no data with the specified name has been cached. It's useful when
    /// you want to check if named data exists before deciding whether to use it or fall back
    /// to the default data.
    /// </remarks>
    public static bool TryGetNamedInputData<T, TInput, TOutput>(string name, out OptimizationInputData<T, TInput, TOutput>? inputData)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Name cannot be empty or whitespace.", nameof(name));

        inputData = null;

        if (_cache.TryGetValue(name, out var cachedData) &&
            cachedData is OptimizationInputData<T, TInput, TOutput> typedData)
        {
            inputData = typedData;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Checks if the default input data is cached.
    /// </summary>
    /// <returns>true if default input data is cached; otherwise, false.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method checks if the main input data that most models use
    /// has been cached. It's useful for determining whether to retrieve cached data
    /// or create new data.
    /// </remarks>
    public static bool IsDefaultDataCached()
    {
        return _cache.ContainsKey(DefaultCacheKey);
    }

    /// <summary>
    /// Checks if input data with a specified name is cached.
    /// </summary>
    /// <param name="name">The name to check for in the cache.</param>
    /// <returns>true if input data with the specified name is cached; otherwise, false.</returns>
    /// <exception cref="ArgumentNullException">Thrown if name is null.</exception>
    /// <exception cref="ArgumentException">Thrown if name is empty or just whitespace.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method checks if custom input data with a specific name
    /// has been cached. It's useful for determining whether to retrieve cached data
    /// or create new data with this name.
    /// </remarks>
    public static bool IsNamedDataCached(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Name cannot be empty or whitespace.", nameof(name));

        return _cache.ContainsKey(name);
    }

    /// <summary>
    /// Removes the default input data from the cache.
    /// </summary>
    /// <returns>true if the default input data was successfully removed; otherwise, false.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method removes the main input data from the cache.
    /// Call this when you're done with the optimization process to free up memory.
    /// </remarks>
    public static bool RemoveDefaultData()
    {
        return _cache.TryRemove(DefaultCacheKey, out _);
    }

    /// <summary>
    /// Removes named input data from the cache.
    /// </summary>
    /// <param name="name">The name of the input data to remove.</param>
    /// <returns>true if input data with the specified name was successfully removed; otherwise, false.</returns>
    /// <exception cref="ArgumentNullException">Thrown if name is null.</exception>
    /// <exception cref="ArgumentException">Thrown if name is empty or just whitespace.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method removes custom input data with a specific name
    /// from the cache. Call this when you're done with the experiment that used this
    /// data to free up memory.
    /// </remarks>
    public static bool RemoveNamedData(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Name cannot be empty or whitespace.", nameof(name));

        return _cache.TryRemove(name, out _);
    }

    /// <summary>
    /// Clears all cached input data, including the default and any named entries.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method removes all data from the cache, freeing up memory.
    /// Use this when you're done with all optimization processes and want to clean up.
    /// </remarks>
    public static void ClearCache()
    {
        _cache.Clear();
    }

    /// <summary>
    /// Gets the total number of entries in the cache, including the default and any named entries.
    /// </summary>
    /// <returns>The count of cached items.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many different datasets are stored in the cache.
    /// It's useful for monitoring cache usage and ensuring that the cache is being properly cleaned up.
    /// </remarks>
    public static int GetCacheSize()
    {
        return _cache.Count;
    }

    /// <summary>
    /// Lists all names of cached input data sets.
    /// </summary>
    /// <returns>An array of cache entry names.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method provides a list of all the names of data sets
    /// currently stored in the cache. It's useful for seeing what data is available
    /// or for diagnostic purposes.
    /// </remarks>
    public static string[] GetCachedDataNames()
    {
        return [.. _cache.Keys];
    }
}