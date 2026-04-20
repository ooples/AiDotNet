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
        var span = weights.Data.Span;
        var rng = Random;

        // Fast path: avoid NumOps.FromDouble virtual dispatch for double/float
        if (typeof(T) == typeof(double))
        {
            for (int i = 0; i < span.Length; i++)
            {
                double value = rng.NextDouble() * 2 * limit - limit;
                span[i] = System.Runtime.CompilerServices.Unsafe.As<double, T>(ref value);
            }
            return;
        }

        if (typeof(T) == typeof(float))
        {
            for (int i = 0; i < span.Length; i++)
            {
                float value = (float)(rng.NextDouble() * 2 * limit - limit);
                span[i] = System.Runtime.CompilerServices.Unsafe.As<float, T>(ref value);
            }
            return;
        }

        for (int i = 0; i < span.Length; i++)
        {
            span[i] = NumOps.FromDouble(rng.NextDouble() * 2 * limit - limit);
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
        var clipBound = 2.0 * stddev;
        var span = weights.Data.Span;

        if (typeof(T) == typeof(double))
        {
            var rawArr = (double[])(object)weights.GetDataArray();
            XavierFillDouble(rawArr, 0, weights.Length, stddev, clipBound);
            return;
        }

        if (typeof(T) == typeof(float))
        {
            var rawArr = (float[])(object)weights.GetDataArray();
            XavierFillFloat(rawArr, 0, weights.Length, stddev, clipBound);
            return;
        }

        for (int i = 0; i < span.Length; i++)
        {
            double value;
            do { value = SampleGaussian(0, stddev); }
            while (Math.Abs(value) > clipBound);
            span[i] = NumOps.FromDouble(value);
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
        var span = weights.Data.Span;
        var rng = Random;

        if (typeof(T) == typeof(double))
        {
            for (int i = 0; i < span.Length; i++)
            {
                double value = rng.NextDouble() * 2 * limit - limit;
                span[i] = System.Runtime.CompilerServices.Unsafe.As<double, T>(ref value);
            }
            return;
        }

        if (typeof(T) == typeof(float))
        {
            for (int i = 0; i < span.Length; i++)
            {
                float value = (float)(rng.NextDouble() * 2 * limit - limit);
                span[i] = System.Runtime.CompilerServices.Unsafe.As<float, T>(ref value);
            }
            return;
        }

        for (int i = 0; i < span.Length; i++)
            span[i] = NumOps.FromDouble(rng.NextDouble() * 2 * limit - limit);
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
        var span = weights.Data.Span;

        if (typeof(T) == typeof(double))
        {
            for (int i = 0; i < span.Length; i++)
            {
                double value = SampleGaussian(0, stddev);
                span[i] = System.Runtime.CompilerServices.Unsafe.As<double, T>(ref value);
            }
            return;
        }

        if (typeof(T) == typeof(float))
        {
            for (int i = 0; i < span.Length; i++)
            {
                float value = (float)SampleGaussian(0, stddev);
                span[i] = System.Runtime.CompilerServices.Unsafe.As<float, T>(ref value);
            }
            return;
        }

        for (int i = 0; i < span.Length; i++)
            span[i] = NumOps.FromDouble(SampleGaussian(0, stddev));
    }

    /// <summary>
    /// Initializes biases to zero (common default).
    /// </summary>
    /// <param name="biases">The biases tensor to initialize.</param>
    protected void ZeroInitializeBiases(Tensor<T> biases)
    {
        biases.AsWritableSpan().Clear();
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

    /// <summary>
    /// Fills a span with <c>N(0, stddev)</c> samples clipped to ±<paramref name="clipBound"/>,
    /// using a paired Box-Muller transform that produces two samples per pair of uniform
    /// draws — halves the <see cref="Math.Log"/>/<see cref="Math.Sqrt"/> call count vs.
    /// calling <see cref="SampleGaussian"/> per element.
    /// </summary>
    /// <remarks>
    /// Replaces the per-element <c>while (Math.Abs(value) &gt; clipBound) do ...</c>
    /// rejection loop which was the dominant cost of DiT-XL lazy weight init (each
    /// block's Dense / SelfAttention layer paid 1–30 s of RNG overhead on first
    /// forward). Rejection rate at 2σ is ~5 %, so in the common case each iteration
    /// produces two usable samples with one log + one sqrt + one sin + one cos + two
    /// multiplies. The inner loop is a tight unvirtualized local function so JIT can
    /// keep everything in registers and auto-vectorize the clip check.
    /// </remarks>
    private void XavierFillDouble(double[] dst, int offset, int length, double stddev, double clipBound)
    {
        if (length == 0) return;

        const int ParallelThreshold = 1 << 18; // 256K doubles ≈ 2MB
        int cores = Math.Max(1, Environment.ProcessorCount);

        if (length < ParallelThreshold || cores == 1)
        {
            FillChunkDouble(dst.AsSpan(offset, length), stddev, clipBound, Random);
            return;
        }

        // For large tensors (typical DiT-XL hidden×4 ≈ 100M elements), partition
        // across cores so init amortizes over the thread pool instead of running
        // single-threaded. Pre-seed per-chunk RNGs from the master so the parallel
        // work remains deterministic relative to the master seed. System.Random
        // is NOT thread-safe, so we MUST use per-thread instances.
        int chunkSize = (length + cores - 1) / cores;
        var seeds = new int[cores];
        for (int c = 0; c < cores; c++) seeds[c] = Random.Next();

        System.Threading.Tasks.Parallel.For(0, cores, c =>
        {
            int chunkStart = c * chunkSize;
            int chunkEnd = Math.Min(chunkStart + chunkSize, length);
            if (chunkStart >= chunkEnd) return;
            var chunkRng = new Random(seeds[c]);
            FillChunkDouble(dst.AsSpan(offset + chunkStart, chunkEnd - chunkStart), stddev, clipBound, chunkRng);
        });
    }

    /// <summary>
    /// Sequential Box-Muller fill of a span — inner helper used by both the
    /// sequential fast path and the parallel chunk workers.
    /// </summary>
    private static void FillChunkDouble(Span<double> dst, double stddev, double clipBound, Random rng)
    {
        double z1 = 0;
        bool havePending = false;

        for (int i = 0; i < dst.Length; i++)
        {
            double sample;
            while (true)
            {
                if (havePending)
                {
                    sample = z1;
                    havePending = false;
                }
                else
                {
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = rng.NextDouble();
                    double r = Math.Sqrt(-2.0 * Math.Log(u1));
                    double theta = 2.0 * Math.PI * u2;
                    sample = r * Math.Sin(theta);
                    z1 = r * Math.Cos(theta);
                    havePending = true;
                }
                sample *= stddev;
                if (!(sample > clipBound) && !(sample < -clipBound))
                {
                    dst[i] = sample;
                    break;
                }
                havePending = false;
            }
        }
    }

    /// <summary>
    /// Float variant of <see cref="XavierFillDouble"/>. Uses double-precision
    /// Box-Muller internally (accuracy matters more than the tiny cost) and
    /// narrows to float on store.
    /// </summary>
    private void XavierFillFloat(float[] dst, int offset, int length, double stddev, double clipBound)
    {
        if (length == 0) return;

        const int ParallelThreshold = 1 << 18;
        int cores = Math.Max(1, Environment.ProcessorCount);

        if (length < ParallelThreshold || cores == 1)
        {
            FillChunkFloat(dst.AsSpan(offset, length), stddev, clipBound, Random);
            return;
        }

        int chunkSize = (length + cores - 1) / cores;
        var seeds = new int[cores];
        for (int c = 0; c < cores; c++) seeds[c] = Random.Next();

        System.Threading.Tasks.Parallel.For(0, cores, c =>
        {
            int chunkStart = c * chunkSize;
            int chunkEnd = Math.Min(chunkStart + chunkSize, length);
            if (chunkStart >= chunkEnd) return;
            var chunkRng = new Random(seeds[c]);
            FillChunkFloat(dst.AsSpan(offset + chunkStart, chunkEnd - chunkStart), stddev, clipBound, chunkRng);
        });
    }

    private static void FillChunkFloat(Span<float> dst, double stddev, double clipBound, Random rng)
    {
        double z1 = 0;
        bool havePending = false;

        for (int i = 0; i < dst.Length; i++)
        {
            double sample;
            while (true)
            {
                if (havePending)
                {
                    sample = z1;
                    havePending = false;
                }
                else
                {
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = rng.NextDouble();
                    double r = Math.Sqrt(-2.0 * Math.Log(u1));
                    double theta = 2.0 * Math.PI * u2;
                    sample = r * Math.Sin(theta);
                    z1 = r * Math.Cos(theta);
                    havePending = true;
                }
                sample *= stddev;
                if (!(sample > clipBound) && !(sample < -clipBound))
                {
                    dst[i] = (float)sample;
                    break;
                }
                havePending = false;
            }
        }
    }
}
