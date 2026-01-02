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
        var result = new Tensor<T>(noise.Shape);
        var noiseSpan = noise.AsSpan();
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Multiply(noiseSpan[i], scaleT);
        }

        return result;
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
        var result = new Tensor<T>(signal.Shape);
        var signalSpan = signal.AsSpan();
        var noiseSpan = noise.AsSpan();
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            // noisy = sqrt(alpha_cumprod) * signal + sqrt(1 - alpha_cumprod) * noise
            var scaledSignal = NumOps.Multiply(sqrtAlphaCumprod, signalSpan[i]);
            var scaledNoise = NumOps.Multiply(sqrtOneMinusAlphaCumprod, noiseSpan[i]);
            resultSpan[i] = NumOps.Add(scaledSignal, scaledNoise);
        }

        return result;
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

        var batchSize = timesteps.Length;
        var halfDim = embeddingDim / 2;
        var result = new Tensor<T>(new[] { batchSize, embeddingDim });
        var resultSpan = result.AsWritableSpan();

        var logScale = Math.Log(10000.0) / (halfDim - 1);

        for (int b = 0; b < batchSize; b++)
        {
            var t = timesteps[b];
            var offset = b * embeddingDim;

            for (int i = 0; i < halfDim; i++)
            {
                var freq = Math.Exp(-i * logScale);
                var angle = t * freq;

                resultSpan[offset + i] = NumOps.FromDouble(Math.Sin(angle));
                resultSpan[offset + i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
            }
        }

        return result;
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
        var result = new Vector<T>(embeddingDim);

        var logScale = Math.Log(10000.0) / (halfDim - 1);

        for (int i = 0; i < halfDim; i++)
        {
            var freq = Math.Exp(-i * logScale);
            var angle = timestep * freq;

            result[i] = NumOps.FromDouble(Math.Sin(angle));
            result[i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
        }

        return result;
    }

    /// <summary>
    /// Computes the signal-to-noise ratio (SNR) for a given timestep.
    /// </summary>
    /// <param name="alphaCumprod">The cumulative alpha product at this timestep.</param>
    /// <returns>The SNR value: alpha / (1 - alpha).</returns>
    public static T ComputeSNR(T alphaCumprod)
    {
        var oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaCumprod);
        if (NumOps.ToDouble(oneMinusAlpha) <= 0)
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

        var result = new Tensor<T>(noise1.Shape);
        var span1 = noise1.AsSpan();
        var span2 = noise2.AsSpan();
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            // result = (1 - t) * noise1 + t * noise2
            resultSpan[i] = NumOps.Add(
                NumOps.Multiply(oneMinusT, span1[i]),
                NumOps.Multiply(tVal, span2[i]));
        }

        return result;
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

        // Flatten tensors to compute dot product
        var span1 = noise1.AsSpan();
        var span2 = noise2.AsSpan();

        // Compute norms and dot product
        double norm1 = 0, norm2 = 0, dot = 0;
        for (int i = 0; i < span1.Length; i++)
        {
            var v1 = NumOps.ToDouble(span1[i]);
            var v2 = NumOps.ToDouble(span2[i]);
            norm1 += v1 * v1;
            norm2 += v2 * v2;
            dot += v1 * v2;
        }
        norm1 = Math.Sqrt(norm1);
        norm2 = Math.Sqrt(norm2);

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

        var result = new Tensor<T>(noise1.Shape);
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var v1 = NumOps.ToDouble(span1[i]);
            var v2 = NumOps.ToDouble(span2[i]);
            resultSpan[i] = NumOps.FromDouble(scale1 * v1 + scale2 * v2);
        }

        return result;
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
