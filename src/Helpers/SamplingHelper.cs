namespace AiDotNet.Helpers;

public static class SamplingHelper
{
    private static Random _random = new Random();

    /// <summary>
    /// Performs sampling without replacement.
    /// </summary>
    /// <param name="populationSize">The size of the population to sample from.</param>
    /// <param name="sampleSize">The number of samples to take.</param>
    /// <returns>An array of indices representing the sampled items.</returns>
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
            int index = _random.Next(indices.Count);
            result[i] = indices[index];
            indices.RemoveAt(index);
        }

        return result;
    }

    /// <summary>
    /// Performs sampling with replacement.
    /// </summary>
    /// <param name="populationSize">The size of the population to sample from.</param>
    /// <param name="sampleSize">The number of samples to take.</param>
    /// <returns>An array of indices representing the sampled items.</returns>
    public static int[] SampleWithReplacement(int populationSize, int sampleSize)
    {
        var result = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
        {
            result[i] = _random.Next(populationSize);
        }
        return result;
    }

    /// <summary>
    /// Creates bootstrap samples from the given data.
    /// </summary>
    /// <typeparam name="T">The type of the data.</typeparam>
    /// <param name="data">The original data to sample from.</param>
    /// <param name="numberOfSamples">The number of bootstrap samples to create.</param>
    /// <param name="sampleSize">The size of each bootstrap sample. If null, it will be the same as the original data size.</param>
    /// <returns>A list of bootstrap samples.</returns>
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
    /// Sets the seed for the random number generator.
    /// </summary>
    /// <param name="seed">The seed value.</param>
    public static void SetSeed(int seed)
    {
        _random = new Random(seed);
    }
}