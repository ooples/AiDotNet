namespace AiDotNet.Initialization;

/// <summary>
/// Base class for initialization strategies providing common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This abstract base class provides shared implementation for all initialization strategies,
/// including common helper methods for weight initialization patterns like Xavier/Glorot
/// and He initialization.
/// </para>
/// <para><b>For Beginners:</b> This base class contains the shared code that all initialization
/// strategies need, avoiding duplication and ensuring consistent behavior across different
/// initialization methods.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class InitializationStrategyBase<T> : IInitializationStrategy<T>
{
    /// <summary>
    /// The numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Thread-safe random number generator.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// Initializes a new instance of the <see cref="InitializationStrategyBase{T}"/> class.
    /// </summary>
    protected InitializationStrategyBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Random = RandomHelper.ThreadSafeRandom;
    }

    /// <inheritdoc />
    public abstract bool IsLazy { get; }

    /// <inheritdoc />
    public abstract bool LoadFromExternal { get; }

    /// <inheritdoc />
    public abstract void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize);

    /// <inheritdoc />
    public abstract void InitializeBiases(Tensor<T> biases);

    /// <summary>
    /// Initializes weights using Xavier/Glorot uniform initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Xavier initialization is designed to keep the variance of activations roughly the same
    /// across layers during forward propagation. It works well with sigmoid and tanh activations.
    /// </para>
    /// <para>
    /// Formula: W ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
    /// </para>
    /// </remarks>
    /// <param name="weights">The weights tensor to initialize.</param>
    /// <param name="fanIn">The number of input units (fan-in).</param>
    /// <param name="fanOut">The number of output units (fan-out).</param>
    protected void XavierUniformInitialize(Tensor<T> weights, int fanIn, int fanOut)
    {
        var limit = Math.Sqrt(6.0 / (fanIn + fanOut));

        for (int i = 0; i < weights.Length; i++)
        {
            var value = Random.NextDouble() * 2 * limit - limit;
            weights.Data.Span[i] = NumOps.FromDouble(value);
        }
    }

    /// <summary>
    /// Initializes weights using Xavier/Glorot normal initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Similar to Xavier uniform but samples from a normal distribution instead.
    /// </para>
    /// <para>
    /// Formula: W ~ N(0, sqrt(2/(fan_in + fan_out)))
    /// </para>
    /// </remarks>
    /// <param name="weights">The weights tensor to initialize.</param>
    /// <param name="fanIn">The number of input units (fan-in).</param>
    /// <param name="fanOut">The number of output units (fan-out).</param>
    protected void XavierNormalInitialize(Tensor<T> weights, int fanIn, int fanOut)
    {
        var stddev = Math.Sqrt(2.0 / (fanIn + fanOut));

        for (int i = 0; i < weights.Length; i++)
        {
            var value = SampleGaussian(0, stddev);
            weights.Data.Span[i] = NumOps.FromDouble(value);
        }
    }

    /// <summary>
    /// Initializes weights using He/Kaiming uniform initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// He initialization is designed for ReLU and its variants. It accounts for the fact that
    /// ReLU zeros out half of the values, requiring larger initial weights.
    /// </para>
    /// <para>
    /// Formula: W ~ U(-sqrt(6/fan_in), sqrt(6/fan_in))
    /// </para>
    /// </remarks>
    /// <param name="weights">The weights tensor to initialize.</param>
    /// <param name="fanIn">The number of input units (fan-in).</param>
    protected void HeUniformInitialize(Tensor<T> weights, int fanIn)
    {
        var limit = Math.Sqrt(6.0 / fanIn);

        for (int i = 0; i < weights.Length; i++)
        {
            var value = Random.NextDouble() * 2 * limit - limit;
            weights.Data.Span[i] = NumOps.FromDouble(value);
        }
    }

    /// <summary>
    /// Initializes weights using He/Kaiming normal initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// He normal initialization samples from a normal distribution with variance 2/fan_in.
    /// </para>
    /// <para>
    /// Formula: W ~ N(0, sqrt(2/fan_in))
    /// </para>
    /// </remarks>
    /// <param name="weights">The weights tensor to initialize.</param>
    /// <param name="fanIn">The number of input units (fan-in).</param>
    protected void HeNormalInitialize(Tensor<T> weights, int fanIn)
    {
        var stddev = Math.Sqrt(2.0 / fanIn);

        for (int i = 0; i < weights.Length; i++)
        {
            var value = SampleGaussian(0, stddev);
            weights.Data.Span[i] = NumOps.FromDouble(value);
        }
    }

    /// <summary>
    /// Initializes biases to zero (common default).
    /// </summary>
    /// <param name="biases">The biases tensor to initialize.</param>
    protected void ZeroInitializeBiases(Tensor<T> biases)
    {
        var zero = NumOps.Zero;
        for (int i = 0; i < biases.Length; i++)
        {
            biases.Data.Span[i] = zero;
        }
    }

    /// <summary>
    /// Samples a value from a Gaussian (normal) distribution using the Box-Muller transform.
    /// </summary>
    /// <param name="mean">The mean of the distribution.</param>
    /// <param name="stddev">The standard deviation of the distribution.</param>
    /// <returns>A sample from the specified Gaussian distribution.</returns>
    protected double SampleGaussian(double mean, double stddev)
    {
        // Box-Muller transform
        var u1 = 1.0 - Random.NextDouble(); // Avoid log(0)
        var u2 = Random.NextDouble();
        var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stddev * randStdNormal;
    }
}
