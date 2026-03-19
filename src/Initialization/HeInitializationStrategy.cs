namespace AiDotNet.Initialization;

/// <summary>
/// He/Kaiming initialization strategy for ReLU-family activations.
/// </summary>
/// <remarks>
/// <para>
/// He initialization accounts for the fact that ReLU zeros out half the values,
/// requiring larger initial weights to maintain variance through the network.
/// This is the recommended strategy for networks using ReLU, Leaky ReLU, GELU, or SiLU.
/// </para>
/// <para><b>For Beginners:</b> Use this when your network uses ReLU or similar activations
/// (which is most modern networks). It's the PyTorch default for convolutional layers.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class HeInitializationStrategy<T> : InitializationStrategyBase<T>
{
    private readonly bool _useNormal;

    /// <summary>
    /// Creates a He initialization strategy.
    /// </summary>
    /// <param name="useNormal">If true, use normal distribution; if false, use uniform.</param>
    public HeInitializationStrategy(bool useNormal = true)
    {
        _useNormal = useNormal;
    }

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        if (_useNormal)
            HeNormalInitialize(weights, inputSize);
        else
            HeUniformInitialize(weights, inputSize);
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
