namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a caching mechanism for storing and retrieving optimization step data during model training.
/// </summary>
/// <remarks>
/// This interface provides methods to store, retrieve, and clear intermediate calculation results
/// during the training process of machine learning models. Caching these results can significantly
/// improve performance by avoiding redundant calculations.
/// 
/// <b>For Beginners:</b> Think of model caching like saving your progress in a video game.
/// 
/// When training machine learning models, the computer performs many calculations in steps:
/// - These calculations can be time-consuming and resource-intensive
/// - By saving (caching) the results of each step, we avoid having to redo calculations
/// - This makes the training process much faster, especially when:
///   * You're experimenting with different model settings
///   * You need to pause and resume training
///   * You want to analyze intermediate results
/// 
/// For example, imagine you're baking a complex cake that requires multiple stages:
/// - Instead of starting from scratch each time you make a mistake
/// - You could save the batter at different stages
/// - If something goes wrong, you can go back to a saved point rather than starting over
/// 
/// This interface provides the methods needed to implement this "save progress" functionality
/// for machine learning models.
/// </remarks>
/// <typeparam name="T">The numeric data type used in the optimization calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ModelCache")]
public interface IModelCache<T, TInput, TOutput>
{
    /// <summary>
    /// Retrieves previously cached optimization step data associated with the specified key.
    /// </summary>
    /// <remarks>
    /// This method looks up and returns cached calculation results from a previous optimization step.
    /// If no data exists for the given key, it returns null.
    /// 
    /// <b>For Beginners:</b> This is like opening a saved file from your computer.
    /// 
    /// When training a machine learning model:
    /// - The model goes through many steps of calculations
    /// - Each step's results can be saved with a unique name (the key)
    /// - This method lets you retrieve those saved results using that name
    /// 
    /// For example:
    /// - You might save the model's state after every 100 training iterations
    /// - Later, you can retrieve the state from iteration 500 by using its key
    /// - If you request data that hasn't been saved, you'll get null (nothing)
    /// 
    /// This is useful when you want to:
    /// - Compare the model's performance at different stages
    /// - Resume training from a specific point
    /// - Analyze how the model evolved during training
    /// </remarks>
    /// <param name="key">A unique identifier for the cached data you want to retrieve.</param>
    /// <returns>The cached optimization step data if found; otherwise, null.</returns>
    OptimizationStepData<T, TInput, TOutput>? GetCachedStepData(string key);

    /// <summary>
    /// Stores optimization step data in the cache with the specified key.
    /// </summary>
    /// <remarks>
    /// This method saves the current state and calculations from an optimization step
    /// so they can be retrieved later, avoiding the need to recalculate them.
    /// 
    /// <b>For Beginners:</b> This is like saving your work to a file on your computer.
    /// 
    /// During model training:
    /// - The model performs complex calculations at each step
    /// - This method lets you save those calculations with a name (the key)
    /// - Later, you can retrieve these saved calculations using that name
    /// 
    /// For example:
    /// - You might save the model's state after processing each batch of data
    /// - You could use keys like "batch_1", "batch_2", etc.
    /// - If you save with a key that already exists, the new data typically replaces the old data
    /// 
    /// This is useful for:
    /// - Checkpointing: saving progress so you can resume if something crashes
    /// - Efficiency: avoiding repeating expensive calculations
    /// - Analysis: keeping track of how your model changes during training
    /// </remarks>
    /// <param name="key">A unique identifier to associate with this cached data.</param>
    /// <param name="stepData">The optimization step data to cache.</param>
    void CacheStepData(string key, OptimizationStepData<T, TInput, TOutput> stepData);

    /// <summary>
    /// Removes all cached optimization step data.
    /// </summary>
    /// <remarks>
    /// This method deletes all previously cached optimization data, freeing up memory
    /// and ensuring that future retrievals will start fresh.
    ///
    /// <b>For Beginners:</b> This is like emptying your recycle bin or clearing your browser cache.
    ///
    /// Sometimes you need to start fresh:
    /// - When you've made significant changes to your model
    /// - When cached data is no longer relevant
    /// - When you need to free up memory
    ///
    /// For example:
    /// - After completing a full training run, you might clear the cache
    /// - Before starting training with a new dataset, you'd clear old cached results
    /// - If you change your model's structure, old cached calculations become invalid
    ///
    /// Clearing the cache:
    /// - Frees up memory that was being used to store cached data
    /// - Ensures you don't accidentally use outdated calculations
    /// - Gives you a clean slate for a new training session
    /// </remarks>
    void ClearCache();

    /// <summary>
    /// Generates a deterministic cache key based on the solution model and input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a deterministic identifier (key) based on the current model state and input data.
    /// The same inputs and model state will always produce the same key, allowing consistent caching
    /// across process restarts and different machines.
    /// </para>
    /// <para>
    /// <b>Implementation Requirements:</b>
    /// - Must use deterministic hashing (e.g., SHA-256) instead of GetHashCode()
    /// - Must serialize parameters in a stable, ordered format
    /// - Must handle null values consistently
    /// - Must use culture-invariant string formatting for numbers
    /// - Keys must remain valid across process restarts
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like creating a unique file name based on the contents
    /// that stays the same forever, even if you restart your computer or run the program again.
    /// </para>
    /// <para>
    /// When training a model:
    /// - Each combination of model parameters and input data produces different results
    /// - This method generates a unique "fingerprint" for each combination using cryptographic hashing
    /// - The fingerprint is used to save and retrieve cached results persistently
    /// </para>
    /// <para>
    /// For example:
    /// - Two identical models with identical inputs will get the same key, always
    /// - The cached result can be retrieved using this key instead of recalculating
    /// - This saves time by avoiding redundant calculations
    /// - Persisted caches remain valid even after restarting the application
    /// </para>
    /// </remarks>
    /// <param name="solution">The model solution to generate a key for.</param>
    /// <param name="inputData">The input data to include in the key generation.</param>
    /// <returns>A deterministic string key for caching (typically a hex-encoded cryptographic hash).</returns>
    string GenerateCacheKey(IFullModel<T, TInput, TOutput> solution, OptimizationInputData<T, TInput, TOutput> inputData);
}
