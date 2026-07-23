namespace AiDotNet.Initialization;

/// <summary>
/// Uniform random initialization with configurable range.
/// </summary>
/// <remarks>
/// <para>
/// Simple uniform initialization in [-bound, bound]. Useful as a baseline
/// or when you need fine control over the initialization range.
/// </para>
/// <para><b>For Beginners:</b> This initializes weights with random values spread
/// evenly across a range. The default range [-0.05, 0.05] is a common starting point.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class UniformInitializationStrategy<T> : InitializationStrategyBase<T>
{
    private readonly double _bound;

    /// <summary>
    /// Creates a uniform initialization strategy.
    /// </summary>
    /// <param name="bound">Half-range: weights sampled from U(-bound, bound). Default: 0.05.</param>
    public UniformInitializationStrategy(double bound = 0.05)
    {
        _bound = bound;
    }

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        var span = weights.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = NumOps.FromDouble((Random.NextDouble() * 2.0 - 1.0) * _bound);
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
