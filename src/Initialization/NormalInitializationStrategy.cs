namespace AiDotNet.Initialization;

/// <summary>
/// Initializes weights from a normal distribution with a configurable mean and standard deviation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NormalInitializationStrategy<T> : InitializationStrategyBase<T>
{
    private readonly double _mean;
    private readonly double _standardDeviation;

    /// <summary>
    /// Creates a normal initialization strategy.
    /// </summary>
    /// <param name="mean">The mean of the normal distribution.</param>
    /// <param name="standardDeviation">The positive standard deviation of the normal distribution.</param>
    public NormalInitializationStrategy(double mean = 0.0, double standardDeviation = 0.02)
        : this(rng: null, mean, standardDeviation)
    {
    }

    /// <summary>
    /// Creates a normal initialization strategy using the supplied random source.
    /// </summary>
    /// <param name="rng">The random source, or <c>null</c> to use the framework default.</param>
    /// <param name="mean">The mean of the normal distribution.</param>
    /// <param name="standardDeviation">The positive standard deviation of the normal distribution.</param>
    public NormalInitializationStrategy(Random? rng, double mean = 0.0, double standardDeviation = 0.02)
        : base(rng)
    {
        if (double.IsNaN(mean) || double.IsInfinity(mean))
        {
            throw new ArgumentOutOfRangeException(nameof(mean), mean, "Mean must be finite.");
        }
        if (standardDeviation <= 0.0 || double.IsNaN(standardDeviation) || double.IsInfinity(standardDeviation))
        {
            throw new ArgumentOutOfRangeException(
                nameof(standardDeviation), standardDeviation, "Standard deviation must be positive and finite.");
        }

        _mean = mean;
        _standardDeviation = standardDeviation;
    }

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override IInitializationStrategy<T> WithSeededRandom(Random rng)
        => new NormalInitializationStrategy<T>(rng, _mean, _standardDeviation);

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        var span = weights.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = NumOps.FromDouble(SampleGaussian(_mean, _standardDeviation));
        }
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
