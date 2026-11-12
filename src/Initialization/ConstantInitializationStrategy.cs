namespace AiDotNet.Initialization;

/// <summary>
/// Constant initialization strategy that sets every weight to one value and every bias to zero.
/// </summary>
/// <remarks>
/// <para>
/// This is <c>nn.init.constant_(w, value)</c>: the starting point of parameters that a paper defines as a constant, such
/// as a normalization scale of one or a learnable attention temperature of one.
/// </para>
/// <para><b>For Beginners:</b> Most weights start random so that units learn different things. A few parameters are
/// meant to start at a known value instead, like a scale that should begin as "leave the input unchanged" (1). This
/// strategy starts every weight at that value.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ConstantInitializationStrategy<T> : InitializationStrategyBase<T>
{
    private readonly double _value;

    /// <summary>
    /// Initializes a new instance that fills weights with <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value every weight starts at; must be finite.</param>
    public ConstantInitializationStrategy(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentOutOfRangeException(nameof(value), value, "The constant must be finite.");
        _value = value;
    }

    /// <summary>Gets the value every weight starts at.</summary>
    public double Value => _value;

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        weights.AsWritableSpan().Fill(NumOps.FromDouble(_value));
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
