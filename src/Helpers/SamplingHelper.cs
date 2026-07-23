namespace AiDotNet.Helpers;

/// <summary>
/// Provides methods for sampling data, which is essential for many AI and machine learning techniques.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Sampling is like picking random items from a collection. This is important in AI
/// for creating training sets, validation sets, and implementing techniques like bootstrapping
/// that help improve model accuracy and reliability.
/// </remarks>
public static class SamplingHelper
{
    /// <summary>
    /// Seeded random instance for reproducible sampling. When null, uses thread-safe random from RandomHelper.
    /// </summary>
    private static Random? _seededRandom;

    /// <summary>
    /// Gets the random number generator used for all sampling operations.
    /// Uses thread-safe random by default, or a seeded instance if SetSeed was called.
    /// </summary>
    private static Random CurrentRandom => _seededRandom ?? RandomHelper.ThreadSafeRandom;

    /// <summary>
    /// Performs sampling without replacement, meaning once an item is selected, 
    /// it cannot be selected again.
    /// </summary>
    /// <param name="populationSize">The size of the population to sample from.</param>
    /// <param name="sampleSize">The number of samples to take.</param>
    /// <returns>An array of indices representing the sampled items.</returns>
    /// <exception cref="ArgumentException">Thrown when sample size is greater than population size.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> Think of this like drawing lottery numbers where each ball can only be 
    /// drawn once. For example, if you have 100 data points and need a random subset of 10,
    /// this method ensures you get 10 different data points.
    /// </remarks>
    public static int[] SampleWithoutReplacement(int populationSize, int sampleSize)
    {
        if (sampleSize > populationSize)
            throw new ArgumentException("Sample size cannot be greater than population size.");

        var indices = new List<int>(populationSize);
        for (int i = 0; i < populationSize; i++)
            indices.Add(i);

        var result = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
        {
            int index = CurrentRandom.Next(indices.Count);
            result[i] = indices[index];
            indices.RemoveAt(index);
        }

        return result;
    }

    /// <summary>
    /// Performs sampling with replacement, meaning the same item can be selected multiple times.
    /// </summary>
    /// <param name="populationSize">The size of the population to sample from.</param>
    /// <param name="sampleSize">The number of samples to take.</param>
    /// <returns>An array of indices representing the sampled items.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is like rolling a die multiple times - you can get the same number 
    /// more than once. In data terms, if you have 100 data points and need 10 samples, 
    /// some data points might be selected multiple times while others might not be selected at all.
    /// 
    /// This approach is useful for techniques like bootstrapping, where repeated sampling helps
    /// estimate the reliability of your model.
    /// </remarks>
    public static int[] SampleWithReplacement(int populationSize, int sampleSize)
    {
        var result = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
        {
            result[i] = CurrentRandom.Next(populationSize);
        }
        return result;
    }

    /// <summary>
    /// Creates bootstrap samples from the given data, which are random samples with replacement
    /// used for estimating statistical properties.
    /// </summary>
    /// <typeparam name="T">The type of the data elements.</typeparam>
    /// <param name="data">The original data array to sample from.</param>
    /// <param name="numberOfSamples">The number of bootstrap samples to create.</param>
    /// <param name="sampleSize">The size of each bootstrap sample. If null, it will be the same as the original data size.</param>
    /// <returns>A list of bootstrap samples, where each sample is an array of data elements.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Bootstrapping is a powerful technique where you create multiple "synthetic" 
    /// datasets by randomly sampling from your original data (with replacement).
    /// 
    /// For example, if you have 100 data points:
    /// 1. You might create 50 different bootstrap samples
    /// 2. Each sample contains 100 randomly selected data points (some repeated, some missing)
    /// 3. You can train 50 different models on these samples
    /// 4. The variation in these models helps you understand how reliable your predictions are
    /// 
    /// This is especially useful when you have limited data but need to understand the uncertainty
    /// in your model's predictions.
    /// </remarks>
    public static List<T[]> CreateBootstrapSamples<T>(T[] data, int numberOfSamples, int? sampleSize = null)
    {
        int actualSampleSize = sampleSize ?? data.Length;
        var samples = new List<T[]>();

        for (int i = 0; i < numberOfSamples; i++)
        {
            var sampleIndices = SampleWithReplacement(data.Length, actualSampleSize);
            var sample = sampleIndices.Select(index => data[index]).ToArray();
            samples.Add(sample);
        }

        return samples;
    }

    /// <summary>
    /// Sets the seed for the random number generator to ensure reproducible results.
    /// </summary>
    /// <param name="seed">The seed value to initialize the random number generator.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Random number generators aren't truly random - they follow mathematical
    /// formulas that produce numbers that appear random. The "seed" is the starting point for
    /// this formula.
    ///
    /// Setting a specific seed means you'll get the same sequence of "random" numbers every time.
    /// This is crucial in AI/ML when you want your experiments to be reproducible - so you can
    /// get the same results when you run your code again, or when someone else runs your code.
    ///
    /// For example, setting seed=42 before training a model ensures that random operations like
    /// data shuffling happen the same way each time.
    ///
    /// Note: Setting a seed overrides the thread-safe behavior. Call ClearSeed() to restore
    /// thread-safe random generation.
    /// </remarks>
    public static void SetSeed(int seed)
    {
        _seededRandom = RandomHelper.CreateSeededRandom(seed);
    }

    /// <summary>
    /// Clears the seed and restores thread-safe random number generation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> After calling SetSeed for reproducible experiments, you can call this
    /// method to go back to using the default thread-safe random generation.
    /// </remarks>
    public static void ClearSeed()
    {
        _seededRandom = null;
    }
}
