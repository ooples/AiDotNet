namespace AiDotNet.Initialization;

/// <summary>
/// Eager initialization strategy that initializes weights immediately on construction.
/// </summary>
/// <remarks>
/// <para>
/// This is the traditional initialization approach where weights are allocated and
/// initialized during layer construction. This ensures all weights are ready before
/// any training or inference begins.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EagerInitializationStrategy<T> : InitializationStrategyBase<T>
{
    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        XavierNormalInitialize(weights, inputSize, outputSize);
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
