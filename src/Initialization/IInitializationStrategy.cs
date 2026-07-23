namespace AiDotNet.Initialization;

/// <summary>
/// Defines a strategy for initializing neural network layer parameters.
/// </summary>
/// <remarks>
/// <para>
/// This interface allows control over when and how layer weights are initialized.
/// Different strategies can be used for different use cases:
/// - Lazy: Defer initialization until first Forward() call (fast construction)
/// - Eager: Initialize immediately on construction (current behavior)
/// - FromFile: Load weights from a file instead of random initialization
/// - Zero: Initialize all weights to zero (useful for testing)
/// </para>
/// <para><b>For Beginners:</b> This controls how the network sets up its initial weights.
///
/// Different strategies have different trade-offs:
/// - Lazy initialization makes network construction fast (good for tests)
/// - Eager initialization is the traditional approach (slightly slower construction)
/// - FromFile loads pre-trained weights (for transfer learning)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("InitializationStrategy")]
public interface IInitializationStrategy<T>
{
    /// <summary>
    /// Gets a value indicating whether this strategy defers initialization until first use.
    /// </summary>
    /// <value>
    /// <c>true</c> if initialization is deferred until first Forward() call;
    /// <c>false</c> if initialization happens immediately.
    /// </value>
    bool IsLazy { get; }

    /// <summary>
    /// Gets a value indicating whether weights should be loaded from an external source.
    /// </summary>
    /// <value>
    /// <c>true</c> if weights should be loaded from file or other external source;
    /// <c>false</c> if weights should be randomly initialized.
    /// </value>
    bool LoadFromExternal { get; }

    /// <summary>
    /// Initializes the weights tensor with appropriate values.
    /// </summary>
    /// <param name="weights">The weights tensor to initialize.</param>
    /// <param name="inputSize">The number of input features.</param>
    /// <param name="outputSize">The number of output features.</param>
    void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize);

    /// <summary>
    /// Initializes the biases tensor with appropriate values.
    /// </summary>
    /// <param name="biases">The biases tensor to initialize.</param>
    void InitializeBiases(Tensor<T> biases);
}
