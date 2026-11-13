using AiDotNet.Extensions;

namespace AiDotNet.Helpers;

/// <summary>
/// Helper class for noise sampling operations in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This static helper provides common noise sampling operations used throughout diffusion models,
/// ensuring consistent implementations and avoiding code duplication.
/// </para>
/// <para>
/// <b>For Beginners:</b> Diffusion models work by adding and removing noise from data.
/// This helper provides the mathematical operations needed for that process:
/// - Sampling Gaussian (bell-curve) noise
/// - Computing noise schedules
/// - Scaling noise for different timesteps
/// </para>
/// </remarks>
public static class DiffusionNoiseHelper<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Samples Gaussian noise from a standard normal distribution N(0, 1).
    /// </summary>
    /// <param name="shape">The shape of the noise tensor to generate.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A tensor filled with Gaussian noise.</returns>
    /// <remarks>
    /// <para>
    /// Uses the Box-Muller transform to convert uniform random numbers to Gaussian.
    /// The existing RandomHelper is used for thread-safe random number generation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates random "static" like you might see on an old TV.
    /// Each value is drawn from a bell curve (Gaussian distribution) centered at 0.
    /// Most values will be close to 0, with occasional larger positive or negative values.
    /// </para>
    /// </remarks>
    public static Tensor<T> SampleGaussian(int[] shape, int? seed = null)
    {
        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        return SampleGaussian(shape, rng);
    }

    /// <summary>
    /// Samples Gaussian noise using a provided random number generator.
    /// </summary>
    /// <param name="shape">The shape of the noise tensor to generate.</param>
    /// <param name="rng">The random number generator to use.</param>
    /// <returns>A tensor filled with Gaussian noise.</returns>
    public static Tensor<T> SampleGaussian(int[] shape, Random rng)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape must be a non-empty array.", nameof(shape));
        if (rng == null)
            throw new ArgumentNullException(nameof(rng));

        var tensor = new Tensor<T>(shape);
        var span = tensor.AsWritableSpan();

        // Box-Muller transform for normal distribution
        for (int i = 0; i < span.Length; i += 2)
        {
            var (z0, z1) = BoxMullerTransform(rng);
            span[i] = NumOps.FromDouble(z0);
            if (i + 1 < span.Length)
                span[i + 1] = NumOps.FromDouble(z1);
        }

        return tensor;
    }

    /// <summary>
    /// Samples Gaussian noise as a Vector.
    /// </summary>
    /// <param name="length">The length of the vector to generate.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A vector filled with Gaussian noise.</returns>
    public static Vector<T> SampleGaussianVector(int length, int? seed = null)
    {
        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        return SampleGaussianVector(length, rng);
    }

    /// <summary>
    /// Samples Gaussian noise as a Vector using a provided random number generator.
    /// </summary>
    /// <param name="length">The length of the vector to generate.</param>
    /// <param name="rng">The random number generator to use.</param>
    /// <returns>A vector filled with Gaussian noise.</returns>
    public static Vector<T> SampleGaussianVector(int length, Random rng)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");
        if (rng == null)
            throw new ArgumentNullException(nameof(rng));

        var vector = new Vector<T>(length);

        for (int i = 0; i < length; i += 2)
        {
            var (z0, z1) = BoxMullerTransform(rng);
            vector[i] = NumOps.FromDouble(z0);
            if (i + 1 < length)
                vector[i + 1] = NumOps.FromDouble(z1);
        }

        return vector;
    }

    /// <summary>
    /// Scales noise by a given factor.
    /// </summary>
    /// <param name="noise">The noise tensor to scale.</param>
    /// <param name="scale">The scaling factor.</param>
    /// <returns>Scaled noise tensor.</returns>
    public static Tensor<T> ScaleNoise(Tensor<T> noise, double scale)
    {
        var scaleT = NumOps.FromDouble(scale);
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return AiDotNetEngine.Current.TensorMultiplyScalar(noise, scaleT);
    }

    /// <summary>
    /// Adds noise to a signal at a specified timestep using the scheduler's noise schedule.
    /// </summary>
    /// <param name="signal">The clean signal.</param>
    /// <param name="noise">The noise to add.</param>
    /// <param name="sqrtAlphaCumprod">The square root of cumulative alpha at this timestep.</param>
    /// <param name="sqrtOneMinusAlphaCumprod">The square root of (1 - cumulative alpha) at this timestep.</param>
    /// <returns>The noisy signal: sqrt(alpha_cumprod) * signal + sqrt(1 - alpha_cumprod) * noise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how we add noise during training:
    /// - At timestep 0: Almost no noise (mostly signal)
    /// - At timestep T: Almost all noise (almost no signal)
    /// - The alphas control this blend based on timestep
    /// </para>
    /// </remarks>
    public static Tensor<T> AddNoise(Tensor<T> signal, Tensor<T> noise, T sqrtAlphaCumprod, T sqrtOneMinusAlphaCumprod)
    {
        // noisy = sqrt(alpha_cumprod) * signal + sqrt(1 - alpha_cumprod) * noise, in the signal's shape
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var scaledSignal = engine.TensorMultiplyScalar(signal, sqrtAlphaCumprod);
        var scaledNoise = engine.TensorMultiplyScalar(noise.Reshape(signal._shape), sqrtOneMinusAlphaCumprod);
        return engine.TensorAdd(scaledSignal, scaledNoise);
    }

    /// <summary>
    /// Computes sinusoidal timestep embeddings (like in Transformers).
    /// </summary>
    /// <param name="timesteps">Array of timesteps to embed.</param>
    /// <param name="embeddingDim">The dimension of each embedding.</param>
    /// <returns>Tensor of shape [batchSize, embeddingDim] containing the embeddings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts timestep numbers (like 100, 500, 999) into
    /// high-dimensional vectors that the neural network can understand. The sinusoidal
    /// pattern helps the network distinguish between nearby and distant timesteps.
    /// </para>
    /// </remarks>
    public static Tensor<T> ComputeTimestepEmbeddings(int[] timesteps, int embeddingDim)
    {
        if (timesteps == null || timesteps.Length == 0)
            throw new ArgumentException("Timesteps must be a non-empty array.", nameof(timesteps));
        if (embeddingDim <= 0 || embeddingDim % 2 != 0)
            throw new ArgumentException("Embedding dimension must be positive and even.", nameof(embeddingDim));

        var halfDim = embeddingDim / 2;
        var steps = new Tensor<int>((int[])timesteps.Clone(), new[] { timesteps.Length }).Cast<double>();

        // [batch, embeddingDim] = [sin(t * freq) | cos(t * freq)]
        return SinusoidalEmbedding(steps, halfDim);
    }

    /// <summary>
    /// Computes sinusoidal embedding for a single timestep.
    /// </summary>
    /// <param name="timestep">The timestep to embed.</param>
    /// <param name="embeddingDim">The dimension of the embedding.</param>
    /// <returns>Vector of length embeddingDim containing the embedding.</returns>
    public static Vector<T> ComputeTimestepEmbedding(int timestep, int embeddingDim)
    {
        if (embeddingDim <= 0 || embeddingDim % 2 != 0)
            throw new ArgumentException("Embedding dimension must be positive and even.", nameof(embeddingDim));

        var halfDim = embeddingDim / 2;
        var step = new Tensor<double>(new[] { 1 });
        step.SetFlat(0, timestep);

        return SinusoidalEmbedding(step, halfDim).ToVector();
    }

    /// <summary>
    /// Sinusoidal embeddings of the given timesteps: row b is
    /// [sin(t_b * f_0), ..., sin(t_b * f_{h-1}), cos(t_b * f_0), ..., cos(t_b * f_{h-1})] with
    /// f_i = exp(-i * ln(10000) / (h - 1)). The angles are formed in double and only the sines and cosines are
    /// converted to <typeparamref name="T"/>: in a narrow T such as Half, t * f_i is off by up to half a radian.
    /// </summary>
    private static Tensor<T> SinusoidalEmbedding(Tensor<double> steps, int halfDim)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
        var engine = AiDotNetEngine.Current;
        var logScale = Math.Log(10000.0) / (halfDim - 1);
        var exponents = engine.TensorMultiplyScalar(engine.TensorArange<double>(0, halfDim, 1), -logScale);
        var angles = engine.TensorOuter(steps, engine.TensorExp(exponents));
        return engine.TensorConcatenate(new[] { engine.TensorSin(angles), engine.TensorCos(angles) }, 1).Cast<T>();
    }

    /// <summary>
    /// Computes the signal-to-noise ratio (SNR) for a given timestep.
    /// </summary>
    /// <param name="alphaCumprod">The cumulative alpha product at this timestep.</param>
    /// <returns>The SNR value: alpha / (1 - alpha).</returns>
    public static T ComputeSNR(T alphaCumprod)
    {
        var oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaCumprod);
        if (!NumOps.GreaterThan(oneMinusAlpha, NumOps.Zero))
        {
            // Avoid division by zero
            return NumOps.FromDouble(1e10);
        }
        return NumOps.Divide(alphaCumprod, oneMinusAlpha);
    }

    /// <summary>
    /// Linearly interpolates between two noise tensors.
    /// </summary>
    /// <param name="noise1">First noise tensor.</param>
    /// <param name="noise2">Second noise tensor.</param>
    /// <param name="t">Interpolation factor (0 = noise1, 1 = noise2).</param>
    /// <returns>Interpolated noise tensor.</returns>
    public static Tensor<T> LerpNoise(Tensor<T> noise1, Tensor<T> noise2, double t)
    {
        t = MathPolyfill.Clamp(t, 0.0, 1.0);
        var tVal = NumOps.FromDouble(t);
        var oneMinusT = NumOps.FromDouble(1.0 - t);

        // result = (1 - t) * noise1 + t * noise2, in noise1's shape
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorAdd(
            engine.TensorMultiplyScalar(noise1, oneMinusT),
            engine.TensorMultiplyScalar(noise2.Reshape(noise1._shape), tVal));
    }

    /// <summary>
    /// Spherical linear interpolation between two noise tensors.
    /// </summary>
    /// <param name="noise1">First noise tensor.</param>
    /// <param name="noise2">Second noise tensor.</param>
    /// <param name="t">Interpolation factor (0 = noise1, 1 = noise2).</param>
    /// <returns>Spherically interpolated noise tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SLERP is like traveling on the surface of a sphere instead of
    /// cutting through it. For noise interpolation, this often gives smoother transitions
    /// between different random seeds.
    /// </para>
    /// </remarks>
    public static Tensor<T> SlerpNoise(Tensor<T> noise1, Tensor<T> noise2, double t)
    {
        t = MathPolyfill.Clamp(t, 0.0, 1.0);

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;

        // Compute norms and dot product over the flattened tensors
        var flat2 = noise2.Reshape(noise1._shape);
        // A narrow T (Half, BFloat16) is reduced in float: its sums overflow past 65504.
        var (squared1, squared2, dot) = typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16)
            ? Products(engine, noise1.Cast<float>(), flat2.Cast<float>())
            : Products(engine, noise1, flat2);
        double norm1 = Math.Sqrt(squared1);
        double norm2 = Math.Sqrt(squared2);

        // Normalize dot product
        dot = dot / (norm1 * norm2 + 1e-10);
        dot = MathPolyfill.Clamp(dot, -1.0, 1.0);

        // Compute angle
        var theta = Math.Acos(dot);

        // If angle is very small, fall back to linear interpolation
        if (Math.Abs(theta) < 1e-6)
        {
            return LerpNoise(noise1, noise2, t);
        }

        var sinTheta = Math.Sin(theta);
        var scale1 = Math.Sin((1.0 - t) * theta) / sinTheta;
        var scale2 = Math.Sin(t * theta) / sinTheta;

        return engine.TensorAdd(
            engine.TensorMultiplyScalar(noise1, NumOps.FromDouble(scale1)),
            engine.TensorMultiplyScalar(flat2, NumOps.FromDouble(scale2)));
    }

    /// <summary>The sums a.a, b.b and a.b over the flattened tensors.</summary>
    private static (double AA, double BB, double AB) Products<TValue>(IEngine engine, Tensor<TValue> a, Tensor<TValue> b)
    {
        var ops = MathHelper.GetNumericOperations<TValue>();
        return (
            ops.ToDouble(engine.TensorSum(engine.TensorMultiply(a, a))),
            ops.ToDouble(engine.TensorSum(engine.TensorMultiply(b, b))),
            ops.ToDouble(engine.TensorSum(engine.TensorMultiply(a, b))));
    }

    /// <summary>
    /// Box-Muller transform to convert uniform random numbers to Gaussian.
    /// </summary>
    /// <param name="rng">The random number generator.</param>
    /// <returns>A pair of independent standard normal random values.</returns>
    private static (double z0, double z1) BoxMullerTransform(Random rng)
    {
        return (rng.NextGaussian(), rng.NextGaussian());
    }
}
