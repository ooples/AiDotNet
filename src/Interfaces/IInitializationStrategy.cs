namespace AiDotNet.Interfaces;

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
public class LazyInitializationStrategy<T> : IInitializationStrategy<T>
{
    /// <inheritdoc />
    public bool IsLazy => true;

    /// <inheritdoc />
    public bool LoadFromExternal => false;

    /// <inheritdoc />
    public void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        // For lazy strategy, actual initialization is done by the layer when first needed
        // This method provides the default Xavier/Glorot initialization
        var numOps = MathHelper.GetNumericOperations<T>();
        var scale = numOps.Sqrt(numOps.FromDouble(2.0 / (inputSize + outputSize)));
        var scaleDouble = Convert.ToDouble(scale);
        var random = RandomHelper.ThreadSafeRandom;

        for (int i = 0; i < weights.Shape[0]; i++)
        {
            for (int j = 0; j < weights.Shape[1]; j++)
            {
                weights[i, j] = numOps.FromDouble(random.NextDouble() * scaleDouble - scaleDouble / 2);
            }
        }
    }

    /// <inheritdoc />
    public void InitializeBiases(Tensor<T> biases)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < biases.Length; i++)
        {
            biases.Data[i] = numOps.Zero;
        }
    }
}

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
public class EagerInitializationStrategy<T> : IInitializationStrategy<T>
{
    /// <inheritdoc />
    public bool IsLazy => false;

    /// <inheritdoc />
    public bool LoadFromExternal => false;

    /// <inheritdoc />
    public void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var scale = numOps.Sqrt(numOps.FromDouble(2.0 / (inputSize + outputSize)));
        var scaleDouble = Convert.ToDouble(scale);
        var random = RandomHelper.ThreadSafeRandom;

        for (int i = 0; i < weights.Shape[0]; i++)
        {
            for (int j = 0; j < weights.Shape[1]; j++)
            {
                weights[i, j] = numOps.FromDouble(random.NextDouble() * scaleDouble - scaleDouble / 2);
            }
        }
    }

    /// <inheritdoc />
    public void InitializeBiases(Tensor<T> biases)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < biases.Length; i++)
        {
            biases.Data[i] = numOps.Zero;
        }
    }
}

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
public class ZeroInitializationStrategy<T> : IInitializationStrategy<T>
{
    /// <inheritdoc />
    public bool IsLazy => false;

    /// <inheritdoc />
    public bool LoadFromExternal => false;

    /// <inheritdoc />
    public void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < weights.Length; i++)
        {
            weights.Data[i] = numOps.Zero;
        }
    }

    /// <inheritdoc />
    public void InitializeBiases(Tensor<T> biases)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < biases.Length; i++)
        {
            biases.Data[i] = numOps.Zero;
        }
    }
}

/// <summary>
/// Provides factory methods and default instances for initialization strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class InitializationStrategy<T>
{
    /// <summary>
    /// Gets the default lazy initialization strategy.
    /// </summary>
    public static IInitializationStrategy<T> Lazy { get; } = new LazyInitializationStrategy<T>();

    /// <summary>
    /// Gets the default eager initialization strategy.
    /// </summary>
    public static IInitializationStrategy<T> Eager { get; } = new EagerInitializationStrategy<T>();

    /// <summary>
    /// Gets the zero initialization strategy.
    /// </summary>
    public static IInitializationStrategy<T> Zero { get; } = new ZeroInitializationStrategy<T>();
}
