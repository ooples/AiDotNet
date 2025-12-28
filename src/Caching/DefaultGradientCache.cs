global using System.Collections.Concurrent;

namespace AiDotNet.Caching;

/// <summary>
/// Provides a default implementation of the gradient caching mechanism for symbolic models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A gradient cache is like a memory bank that stores pre-calculated mathematical operations
/// that are frequently used during machine learning model training.
/// 
/// In machine learning, "gradients" are calculations that show how much a model's error would change
/// if we slightly adjusted a specific parameter. These calculations can be complex and time-consuming,
/// especially if they need to be repeated many times.
/// 
/// By storing these calculations in a cache (a temporary storage area), we can avoid recalculating the
/// same gradients repeatedly, which makes the training process much faster. Think of it like remembering
/// the answer to a difficult math problem so you don't have to solve it again when you need the same answer later.
/// 
/// This class provides a simple way to store and retrieve these gradient calculations using string keys
/// (like names or identifiers) to look them up quickly.
/// </para>
/// </remarks>
public class DefaultGradientCache<T> : IGradientCache<T>
{
    /// <summary>
    /// The internal dictionary that stores gradient models, allowing concurrent access from multiple threads.
    /// </summary>
    /// <remarks>
    /// ConcurrentDictionary is used to ensure thread safety when multiple operations might access the cache simultaneously.
    /// </remarks>
    private readonly ConcurrentDictionary<string, IGradientModel<T>> _cache = new();

    /// <summary>
    /// Retrieves a cached gradient model using the specified key.
    /// </summary>
    /// <param name="key">The unique identifier for the gradient model.</param>
    /// <returns>The cached gradient model if found; otherwise, null.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method works like looking up a word in a dictionary. You provide a key (like a word),
    /// and it returns the corresponding gradient model (like the definition) if it exists in the cache.
    /// 
    /// If the key doesn't exist in the cache, the method returns null, indicating that the gradient
    /// needs to be calculated from scratch.
    /// </para>
    /// </remarks>
    public IGradientModel<T>? GetCachedGradient(string key)
    {
        if (key == null) throw new ArgumentNullException(nameof(key), "Cache key cannot be null.");

        _cache.TryGetValue(key, out var gradient);
        return gradient;
    }

    /// <summary>
    /// Stores a gradient model in the cache with the specified key.
    /// </summary>
    /// <param name="key">The unique identifier for the gradient model.</param>
    /// <param name="gradient">The gradient model to cache.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method saves a gradient model in the cache so it can be retrieved later.
    /// 
    /// It's like writing down the answer to a complex math problem with a label (the key) so you can
    /// look it up quickly next time instead of solving the problem again.
    /// 
    /// If a gradient with the same key already exists in the cache, it will be replaced with the new one.
    /// </para>
    /// </remarks>
    public void CacheGradient(string key, IGradientModel<T> gradient)
    {
        if (key == null) throw new ArgumentNullException(nameof(key), "Cache key cannot be null.");
        if (gradient == null) throw new ArgumentNullException(nameof(gradient), "Gradient cannot be null.");

        _cache[key] = gradient;
    }

    /// <summary>
    /// Removes all cached gradient models from the cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this method when you want to start fresh, such as when beginning a new training session
    /// or when you've made changes to your model that would invalidate previously cached gradients.
    /// </para>
    /// </remarks>
    public void ClearCache()
    {
        _cache.Clear();
    }
}
