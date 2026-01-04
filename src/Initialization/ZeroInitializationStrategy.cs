namespace AiDotNet.Initialization;

/// <summary>
/// Zero initialization strategy that sets all weights to zero.
/// </summary>
/// <remarks>
/// <para>
/// This strategy initializes all weights to zero. It is primarily useful for testing
/// to ensure deterministic behavior, or for specific network architectures where
/// zero initialization is desired for certain layers.
/// </para>
/// <para><b>Warning:</b> Zero initialization typically should not be used for training
/// as it prevents symmetry breaking and leads to poor learning. Use for testing only.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ZeroInitializationStrategy<T> : InitializationStrategyBase<T>
{
    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        var zero = NumOps.Zero;
        for (int i = 0; i < weights.Length; i++)
        {
            weights.Data[i] = zero;
        }
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
