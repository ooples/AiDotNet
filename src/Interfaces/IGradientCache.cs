namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for storing and retrieving pre-computed gradients to improve performance in machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface defines methods for saving and reusing calculations to make your AI models run faster.
/// 
/// In machine learning, models often need to calculate "gradients" - mathematical directions that show how to 
/// adjust the model to make better predictions. These calculations can be time-consuming, especially for complex models.
/// 
/// Think of a gradient cache like a notebook where you write down answers to difficult math problems:
/// - When you solve a problem, you write down the answer in your notebook
/// - Later, if you need the same answer again, you can just look it up instead of re-solving the problem
/// - This saves you time and effort
/// 
/// The gradient cache works the same way:
/// - When a gradient is calculated, it's stored with a unique name (the "key")
/// - If the same gradient is needed again, it can be retrieved using that name
/// - This avoids repeating expensive calculations
/// 
/// This is especially useful for:
/// - Complex models with many parameters
/// - Models that use the same calculations repeatedly
/// - Training scenarios where speed is important
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("GradientCache")]
public interface IGradientCache<T>
{
    /// <summary>
    /// Retrieves a previously cached gradient using its unique key.
    /// </summary>
    /// <param name="key">The unique identifier for the gradient.</param>
    /// <returns>The cached gradient if found; otherwise, null.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method looks up a saved calculation in the cache.
    /// 
    /// The parameter:
    /// - key: A unique name or identifier for the gradient you want to retrieve
    ///   (like a file name or a label in your notebook)
    /// 
    /// What this method does:
    /// 1. Checks if a gradient with the given key exists in the cache
    /// 2. If found, returns the pre-computed gradient
    /// 3. If not found, returns null (meaning you'll need to calculate it from scratch)
    /// 
    /// The returned ISymbolicModel represents the gradient in a form that can be:
    /// - Evaluated with different input values
    /// - Combined with other mathematical expressions
    /// - Used directly in optimization algorithms
    /// 
    /// Using cached gradients can significantly speed up training because you avoid
    /// repeating the same complex calculations multiple times.
    /// </remarks>
    IGradientModel<T>? GetCachedGradient(string key);

    /// <summary>
    /// Stores a gradient in the cache with a unique key for later retrieval.
    /// </summary>
    /// <param name="key">The unique identifier to associate with this gradient.</param>
    /// <param name="gradient">The gradient to cache.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method saves a calculation result so you can reuse it later.
    /// 
    /// The parameters:
    /// - key: A unique name or identifier you choose for this gradient
    ///   (like naming a file or labeling a page in your notebook)
    /// - gradient: The actual gradient calculation result you want to save
    /// 
    /// What this method does:
    /// 1. Takes the gradient you've calculated
    /// 2. Stores it in memory using the key as its identifier
    /// 3. Makes it available for quick retrieval later
    /// 
    /// It's important to use meaningful, consistent keys so you can easily find
    /// your cached gradients later. For example, you might use keys like:
    /// - "layer1_weights_gradient"
    /// - "output_bias_gradient"
    /// 
    /// If you cache a gradient with a key that already exists in the cache,
    /// the new gradient typically replaces the old one.
    /// </remarks>
    void CacheGradient(string key, IGradientModel<T> gradient);

    /// <summary>
    /// Removes all cached gradients, freeing up memory.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method erases all saved calculations from the cache.
    /// 
    /// What this method does:
    /// 1. Removes all gradients that have been stored in the cache
    /// 2. Frees up the memory they were using
    /// 3. Essentially gives you a "clean slate"
    /// 
    /// You might want to clear the cache when:
    /// - You've made significant changes to your model
    /// - You're starting a new phase of training
    /// - You suspect the cached gradients might be outdated
    /// - You need to free up memory
    /// - You want to measure performance without caching
    /// 
    /// After clearing the cache, any attempt to retrieve a gradient will return null
    /// until new gradients are cached.
    /// </remarks>
    void ClearCache();
}
