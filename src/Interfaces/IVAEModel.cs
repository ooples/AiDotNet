namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for Variational Autoencoder (VAE) models used in latent diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VAEs are used in latent diffusion models to compress images into a lower-dimensional
/// latent space where the diffusion process operates. This makes training and generation
/// much more efficient than operating in pixel space.
/// </para>
/// <para>
/// <b>For Beginners:</b> A VAE is like a very smart image compressor and decompressor.
///
/// How it works:
/// 1. Encoder: Takes a full-size image (e.g., 512x512x3) and compresses it to a small latent (e.g., 64x64x4)
/// 2. Decoder: Takes the small latent and reconstructs a full-size image
/// 3. The compression is lossy but learned to preserve important visual information
///
/// Why use a VAE in diffusion?
/// - Full images are huge (512x512x3 = 786,432 values)
/// - Latents are small (64x64x4 = 16,384 values) - 48x smaller!
/// - Diffusion in latent space is much faster
/// - Quality remains high because the VAE learns what matters
///
/// Different VAE types:
/// - Standard VAE: Original Stable Diffusion VAE, 4 latent channels
/// - Tiny VAE: Faster but lower quality, good for previews
/// - Temporal VAE: Video-aware VAE that handles frame consistency
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> to provide all standard
/// model capabilities (training, saving, loading, gradients, checkpointing, etc.).
/// </para>
/// </remarks>
public interface IVAEModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the number of input channels (image channels).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Typically 3 for RGB images. Could be 1 for grayscale or 4 for RGBA.
    /// </para>
    /// </remarks>
    int InputChannels { get; }

    /// <summary>
    /// Gets the number of latent channels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Standard Stable Diffusion VAEs use 4 latent channels.
    /// Some newer VAEs may use different values (e.g., 16 for certain architectures).
    /// </para>
    /// </remarks>
    int LatentChannels { get; }

    /// <summary>
    /// Gets the spatial downsampling factor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The factor by which the VAE reduces spatial dimensions.
    /// Stable Diffusion uses 8x downsampling, so a 512x512 image becomes 64x64 latents.
    /// </para>
    /// </remarks>
    int DownsampleFactor { get; }

    /// <summary>
    /// Gets the scale factor for latent values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A normalization factor applied to latent values. For Stable Diffusion,
    /// this is 0.18215, which normalizes the latent distribution to unit variance.
    /// </para>
    /// </remarks>
    double LatentScaleFactor { get; }

    /// <summary>
    /// Encodes an image into the latent space.
    /// </summary>
    /// <param name="image">The input image tensor [batch, channels, height, width].</param>
    /// <param name="sampleMode">If true, samples from the latent distribution. If false, returns the mean.</param>
    /// <returns>The latent representation [batch, latentChannels, height/downFactor, width/downFactor].</returns>
    /// <remarks>
    /// <para>
    /// The VAE encoder outputs a distribution (mean and log variance). When sampleMode is true,
    /// we sample from this distribution using the reparameterization trick. When false, we just
    /// return the mean for deterministic encoding.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This compresses the image:
    /// - Input: Full-size image (512x512x3)
    /// - Output: Small latent representation (64x64x4)
    /// - The latent contains all the important information in a compressed form
    /// </para>
    /// </remarks>
    Tensor<T> Encode(Tensor<T> image, bool sampleMode = true);

    /// <summary>
    /// Encodes and returns both mean and log variance (for training).
    /// </summary>
    /// <param name="image">The input image tensor [batch, channels, height, width].</param>
    /// <returns>Tuple of (mean, logVariance) tensors.</returns>
    /// <remarks>
    /// <para>
    /// Used during VAE training where we need both the mean and variance for
    /// computing the KL divergence loss.
    /// </para>
    /// </remarks>
    (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> image);

    /// <summary>
    /// Decodes a latent representation back to image space.
    /// </summary>
    /// <param name="latent">The latent tensor [batch, latentChannels, latentHeight, latentWidth].</param>
    /// <returns>The decoded image [batch, channels, height*downFactor, width*downFactor].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This decompresses the latent back to an image:
    /// - Input: Small latent (64x64x4)
    /// - Output: Full-size image (512x512x3)
    /// - The image looks like the original but with minor differences due to compression
    /// </para>
    /// </remarks>
    Tensor<T> Decode(Tensor<T> latent);

    /// <summary>
    /// Samples from the latent distribution using the reparameterization trick.
    /// </summary>
    /// <param name="mean">The mean of the latent distribution.</param>
    /// <param name="logVariance">The log variance of the latent distribution.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A sample from the distribution: mean + std * epsilon.</returns>
    /// <remarks>
    /// <para>
    /// The reparameterization trick allows gradients to flow through the sampling operation:
    /// z = mean + exp(0.5 * logVariance) * epsilon, where epsilon ~ N(0, 1)
    /// </para>
    /// </remarks>
    Tensor<T> Sample(Tensor<T> mean, Tensor<T> logVariance, int? seed = null);

    /// <summary>
    /// Scales latent values for use in diffusion (applies LatentScaleFactor).
    /// </summary>
    /// <param name="latent">The raw latent from encoding.</param>
    /// <returns>Scaled latent values.</returns>
    /// <remarks>
    /// <para>
    /// Multiplies by LatentScaleFactor to normalize the latent distribution.
    /// This is necessary because VAE latents have a specific variance that
    /// diffusion models expect to be normalized.
    /// </para>
    /// </remarks>
    Tensor<T> ScaleLatent(Tensor<T> latent);

    /// <summary>
    /// Unscales latent values before decoding (inverts LatentScaleFactor).
    /// </summary>
    /// <param name="latent">The scaled latent from diffusion.</param>
    /// <returns>Unscaled latent values ready for decoding.</returns>
    /// <remarks>
    /// <para>
    /// Divides by LatentScaleFactor to reverse the scaling before decoding.
    /// </para>
    /// </remarks>
    Tensor<T> UnscaleLatent(Tensor<T> latent);

    /// <summary>
    /// Gets whether this VAE uses tiling for memory-efficient encoding/decoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Tiling processes the image in overlapping patches to reduce memory usage
    /// when handling large images. Useful for high-resolution generation.
    /// </para>
    /// </remarks>
    bool SupportsTiling { get; }

    /// <summary>
    /// Gets whether this VAE uses slicing for sequential processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Slicing processes the batch one sample at a time to reduce memory.
    /// Trades speed for memory efficiency.
    /// </para>
    /// </remarks>
    bool SupportsSlicing { get; }

    /// <summary>
    /// Enables or disables tiling mode.
    /// </summary>
    /// <param name="enabled">Whether to enable tiling.</param>
    void SetTilingEnabled(bool enabled);

    /// <summary>
    /// Enables or disables slicing mode.
    /// </summary>
    /// <param name="enabled">Whether to enable slicing.</param>
    void SetSlicingEnabled(bool enabled);
}
