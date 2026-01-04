namespace AiDotNet.Initialization;

/// <summary>
/// Lazy initialization strategy that defers weight allocation until first Forward() call.
/// </summary>
/// <remarks>
/// <para>
/// This strategy significantly speeds up network construction by not allocating or
/// initializing weight tensors until they are actually needed. This is particularly
/// useful for tests and when comparing network architectures without actually running them.
/// </para>
/// <para><b>For Beginners:</b> Think of lazy initialization like making dinner reservations
/// versus cooking dinner. The reservation (lazy) is fast - the cooking happens later when you
/// actually need it. This makes creating networks much faster when you just want to inspect
/// them or compare their structures.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LazyInitializationStrategy<T> : InitializationStrategyBase<T>
{
    /// <inheritdoc />
    public override bool IsLazy => true;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        // For lazy strategy, actual initialization is done by the layer when first needed
        // This method provides the default Xavier/Glorot initialization
        XavierNormalInitialize(weights, inputSize, outputSize);
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
