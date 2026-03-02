using AiDotNet.Helpers;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Initializes latent tensors for diffusion generation with various noise strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Provides different strategies for initializing the starting noise latent for diffusion
/// generation. Supports standard Gaussian noise, image-conditioned initialization (img2img),
/// and strength-based partial noise for editing workflows. Ensures consistent shapes and
/// proper scaling for the target scheduler.
/// </para>
/// <para>
/// <b>For Beginners:</b> Before a diffusion model can create an image, it needs a starting
/// point — usually pure random noise. LatentInitializer handles creating this starting noise.
/// For text-to-image, it creates pure noise. For image editing (img2img), it mixes the original
/// image with noise, controlling how much of the original to keep.
/// </para>
/// </remarks>
public class LatentInitializer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _initNoiseSigma;
    private readonly Random _random;

    /// <summary>
    /// Gets the initial noise sigma scaling factor.
    /// </summary>
    public double InitNoiseSigma => _initNoiseSigma;

    /// <summary>
    /// Initializes a new latent initializer.
    /// </summary>
    /// <param name="initNoiseSigma">Noise sigma for initial latent scaling (default: 1.0).</param>
    /// <param name="seed">Random seed for reproducibility (default: null for random).</param>
    public LatentInitializer(double initNoiseSigma = 1.0, int? seed = null)
    {
        _initNoiseSigma = initNoiseSigma;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Creates a pure noise latent for text-to-image generation.
    /// </summary>
    /// <param name="size">Total number of elements in the latent tensor.</param>
    /// <returns>Gaussian noise latent scaled by initNoiseSigma.</returns>
    public Vector<T> CreateNoiseLatent(int size)
    {
        var latent = new Vector<T>(size);
        var sigma = NumOps.FromDouble(_initNoiseSigma);

        for (int i = 0; i < size; i++)
        {
            // Box-Muller transform for Gaussian noise
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            latent[i] = NumOps.Multiply(NumOps.FromDouble(gaussian), sigma);
        }

        return latent;
    }

    /// <summary>
    /// Creates a latent initialized from an encoded image with noise added (img2img).
    /// </summary>
    /// <param name="encodedImage">VAE-encoded image latent.</param>
    /// <param name="strength">Noise strength (0.0 = keep original, 1.0 = pure noise).</param>
    /// <returns>Image-conditioned noisy latent.</returns>
    public Vector<T> CreateImg2ImgLatent(Vector<T> encodedImage, double strength)
    {
        var latent = new Vector<T>(encodedImage.Length);
        var strengthT = NumOps.FromDouble(strength);
        var oneMinusStrength = NumOps.Subtract(NumOps.One, strengthT);
        var sigma = NumOps.FromDouble(_initNoiseSigma);

        for (int i = 0; i < encodedImage.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            var noise = NumOps.Multiply(NumOps.FromDouble(gaussian), sigma);

            latent[i] = NumOps.Add(
                NumOps.Multiply(oneMinusStrength, encodedImage[i]),
                NumOps.Multiply(strengthT, noise));
        }

        return latent;
    }
}
