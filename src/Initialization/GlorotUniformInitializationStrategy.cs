namespace AiDotNet.Initialization;

/// <summary>
/// Glorot and Bengio's uniform initialization (TensorFlow's and Keras's default): weights from
/// U(−√(6 / (fan_in + fan_out)), √(6 / (fan_in + fan_out))) and zero biases.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Random starting weights sized so a signal keeps roughly the same spread as it passes through
/// each layer, forward and backward. It is what the Soft Actor-Critic authors' TensorFlow code starts every network
/// with.
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class GlorotUniformInitializationStrategy<T> : InitializationStrategyBase<T>
{
    /// <summary>Creates the initializer.</summary>
    /// <param name="random">The generator to draw from; a seeded one makes the weights reproducible.</param>
    public GlorotUniformInitializationStrategy(Random? random = null)
        : base(random)
    {
    }

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        double limit = Math.Sqrt(6.0 / (inputSize + outputSize));
        var span = weights.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = NumOps.FromDouble((Random.NextDouble() * 2.0 - 1.0) * limit);
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases) => ZeroInitializeBiases(biases);
}
