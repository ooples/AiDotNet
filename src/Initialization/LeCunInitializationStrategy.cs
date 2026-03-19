namespace AiDotNet.Initialization;

/// <summary>
/// LeCun initialization strategy for SELU activations and self-normalizing networks.
/// </summary>
/// <remarks>
/// <para>
/// LeCun initialization uses variance 1/fan_in, designed for networks with SELU activation
/// that maintain self-normalizing properties through many layers.
/// </para>
/// <para><b>For Beginners:</b> Use this with SELU activation functions for deep networks
/// that automatically maintain stable activations without batch normalization.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LeCunInitializationStrategy<T> : InitializationStrategyBase<T>
{
    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        var stddev = Math.Sqrt(1.0 / inputSize);
        var span = weights.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = NumOps.FromDouble(SampleGaussian(0, stddev));
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
