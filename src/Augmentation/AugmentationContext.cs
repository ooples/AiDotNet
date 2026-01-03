using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.Augmentation;

/// <summary>
/// Provides runtime context for augmentation operations including random state,
/// training mode, and spatial targets.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AugmentationContext<T>
{
    private readonly Random _random;

    /// <summary>
    /// Gets the random number generator for this context.
    /// </summary>
    public Random Random => _random;

    /// <summary>
    /// Gets whether the context is in training mode.
    /// </summary>
    public bool IsTraining { get; }

    /// <summary>
    /// Gets the batch index (if applicable).
    /// </summary>
    public int BatchIndex { get; set; }

    /// <summary>
    /// Gets the sample index within the batch (if applicable).
    /// </summary>
    public int SampleIndex { get; set; }

    /// <summary>
    /// Gets additional metadata for the current augmentation.
    /// </summary>
    public IDictionary<string, object> Metadata { get; }

    /// <summary>
    /// Creates a new augmentation context.
    /// </summary>
    /// <param name="isTraining">Whether the context is in training mode.</param>
    /// <param name="seed">Optional seed for reproducibility.</param>
    public AugmentationContext(bool isTraining = true, int? seed = null)
    {
        IsTraining = isTraining;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        Metadata = new Dictionary<string, object>();
    }

    /// <summary>
    /// Creates a new augmentation context with a provided random instance.
    /// </summary>
    /// <param name="random">The random number generator to use.</param>
    /// <param name="isTraining">Whether the context is in training mode.</param>
    public AugmentationContext(Random random, bool isTraining = true)
    {
        _random = random ?? throw new ArgumentNullException(nameof(random));
        IsTraining = isTraining;
        Metadata = new Dictionary<string, object>();
    }

    /// <summary>
    /// Determines whether an augmentation with the given probability should be applied.
    /// </summary>
    /// <param name="probability">The probability (0.0 to 1.0).</param>
    /// <returns>True if the augmentation should be applied.</returns>
    public bool ShouldApply(double probability)
    {
        if (probability >= 1.0) return true;
        if (probability <= 0.0) return false;
        return _random.NextDouble() < probability;
    }

    /// <summary>
    /// Gets a random value within the specified range.
    /// </summary>
    /// <param name="min">The minimum value (inclusive).</param>
    /// <param name="max">The maximum value (exclusive).</param>
    /// <returns>A random value in the range [min, max).</returns>
    public double GetRandomDouble(double min, double max)
    {
        return min + (_random.NextDouble() * (max - min));
    }

    /// <summary>
    /// Gets a random integer within the specified range.
    /// </summary>
    /// <param name="min">The minimum value (inclusive).</param>
    /// <param name="max">The maximum value (exclusive).</param>
    /// <returns>A random integer in the range [min, max).</returns>
    public int GetRandomInt(int min, int max)
    {
        return _random.Next(min, max);
    }

    /// <summary>
    /// Gets a random boolean with 50% probability.
    /// </summary>
    /// <returns>A random boolean.</returns>
    public bool GetRandomBool()
    {
        return _random.NextDouble() < 0.5;
    }

    /// <summary>
    /// Samples from a Beta distribution (used by Mixup/CutMix).
    /// </summary>
    /// <param name="alpha">The alpha parameter.</param>
    /// <param name="beta">The beta parameter.</param>
    /// <returns>A sample from Beta(alpha, beta).</returns>
    public double SampleBeta(double alpha, double beta)
    {
        // Use the gamma distribution method for sampling beta
        double x = SampleGamma(alpha);
        double y = SampleGamma(beta);
        return x / (x + y);
    }

    /// <summary>
    /// Samples from a Gamma distribution.
    /// </summary>
    /// <param name="shape">The shape parameter.</param>
    /// <returns>A sample from Gamma(shape, 1).</returns>
    private double SampleGamma(double shape)
    {
        // Marsaglia and Tsang's method
        if (shape < 1)
        {
            return SampleGamma(1 + shape) * Math.Pow(_random.NextDouble(), 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);

        while (true)
        {
            double x, v;
            do
            {
                x = SampleStandardNormal();
                v = 1.0 + c * x;
            }
            while (v <= 0);

            v = v * v * v;
            double u = _random.NextDouble();

            if (u < 1.0 - 0.0331 * x * x * x * x)
                return d * v;

            if (Math.Log(u) < 0.5 * x * x + d * (1 - v + Math.Log(v)))
                return d * v;
        }
    }

    /// <summary>
    /// Samples from a Gaussian (normal) distribution.
    /// </summary>
    /// <param name="mean">The mean of the distribution.</param>
    /// <param name="stdDev">The standard deviation of the distribution.</param>
    /// <returns>A sample from N(mean, stdDev^2).</returns>
    public double SampleGaussian(double mean, double stdDev)
    {
        return mean + stdDev * SampleStandardNormal();
    }

    /// <summary>
    /// Samples from a standard normal distribution.
    /// </summary>
    /// <returns>A sample from N(0, 1).</returns>
    private double SampleStandardNormal()
    {
        return _random.NextGaussian();
    }

    /// <summary>
    /// Creates a child context with the same random state but different indices.
    /// </summary>
    /// <param name="sampleIndex">The sample index for the child context.</param>
    /// <returns>A new child context.</returns>
    public AugmentationContext<T> CreateChildContext(int sampleIndex)
    {
        return new AugmentationContext<T>(_random, IsTraining)
        {
            BatchIndex = BatchIndex,
            SampleIndex = sampleIndex
        };
    }
}
