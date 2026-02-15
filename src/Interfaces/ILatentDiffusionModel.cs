namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for latent diffusion models that operate in a compressed latent space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Latent diffusion models are a highly efficient variant of diffusion models that perform
/// the denoising process in a compressed latent space rather than pixel space. This is the
/// architecture behind Stable Diffusion and many other state-of-the-art generative models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Latent diffusion combines the power of diffusion models with the
/// efficiency of autoencoders.
///
/// How it works:
/// 1. A VAE compresses images (512x512) into small latents (64x64)
/// 2. Diffusion happens in this compressed space (much faster!)
/// 3. The VAE decompresses the result back to a full image
///
/// Benefits:
/// - Training is ~50x faster than pixel-space diffusion
/// - Generation is ~50x faster
/// - Quality remains very high
/// - Enables practical high-resolution generation
///
/// Key components:
/// - VAE: Compresses and decompresses images
/// - Noise Predictor (U-Net/DiT): Predicts noise in latent space
/// - Scheduler: Controls the denoising process
/// - Conditioner: Encodes text/images for guided generation
/// </para>
/// <para>
/// This interface extends <see cref="IDiffusionModel{T}"/> with latent-space specific operations.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("LatentDiffusionModel")]
public interface ILatentDiffusionModel<T> : IDiffusionModel<T>
{
    /// <summary>
    /// Gets the VAE model used for encoding and decoding.
    /// </summary>
    IVAEModel<T> VAE { get; }

    /// <summary>
    /// Gets the noise predictor model (U-Net, DiT, etc.).
    /// </summary>
    INoisePredictor<T> NoisePredictor { get; }

    /// <summary>
    /// Gets the conditioning module (optional, for conditioned generation).
    /// </summary>
    IConditioningModule<T>? Conditioner { get; }

    /// <summary>
    /// Gets the number of latent channels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Typically 4 for Stable Diffusion models.
    /// </para>
    /// </remarks>
    int LatentChannels { get; }

    /// <summary>
    /// Gets the default guidance scale for classifier-free guidance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Higher values make generation more closely follow the conditioning.
    /// Typical values: 7.5 for Stable Diffusion, 5.0 for SDXL.
    /// </para>
    /// </remarks>
    double GuidanceScale { get; }

    /// <summary>
    /// Encodes an image into latent space.
    /// </summary>
    /// <param name="image">The input image tensor [batch, channels, height, width].</param>
    /// <param name="sampleMode">Whether to sample from the VAE distribution.</param>
    /// <returns>The latent representation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This compresses an image for processing:
    /// - Input: Full-size image (e.g., 512x512)
    /// - Output: Small latent (e.g., 64x64x4)
    ///
    /// Use sampleMode=true during training for VAE regularization,
    /// and sampleMode=false for deterministic encoding during editing.
    /// </para>
    /// </remarks>
    Tensor<T> EncodeToLatent(Tensor<T> image, bool sampleMode = true);

    /// <summary>
    /// Decodes a latent representation back to an image.
    /// </summary>
    /// <param name="latent">The latent tensor.</param>
    /// <returns>The decoded image tensor [batch, channels, height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This decompresses a latent back to an image:
    /// - Input: Small latent (e.g., 64x64x4)
    /// - Output: Full-size image (e.g., 512x512x3)
    /// </para>
    /// </remarks>
    Tensor<T> DecodeFromLatent(Tensor<T> latent);

    /// <summary>
    /// Generates images from text prompts using classifier-free guidance.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="negativePrompt">Optional negative prompt (what to avoid).</param>
    /// <param name="width">Image width in pixels (should be divisible by VAE downsample factor).</param>
    /// <param name="height">Image height in pixels (should be divisible by VAE downsample factor).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">How closely to follow the prompt (higher = closer).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>The generated image tensor.</returns>
    /// <remarks>
    /// <para>
    /// This is the main text-to-image generation method. It performs:
    /// 1. Encode text prompts to conditioning embeddings
    /// 2. Generate random latent noise
    /// 3. Iteratively denoise with classifier-free guidance
    /// 4. Decode latent to image
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how you generate images from text:
    /// - prompt: What you want ("a cat in a spacesuit")
    /// - negativePrompt: What to avoid ("blurry, low quality")
    /// - guidanceScale: How strictly to follow the prompt (7.5 is typical)
    /// </para>
    /// </remarks>
    Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null);

    /// <summary>
    /// Performs image-to-image generation (style transfer, editing).
    /// </summary>
    /// <param name="inputImage">The input image to transform.</param>
    /// <param name="prompt">The text prompt describing the desired transformation.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">How much to transform (0.0 = no change, 1.0 = full regeneration).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The transformed image tensor.</returns>
    /// <remarks>
    /// <para>
    /// Image-to-image works by:
    /// 1. Encode the input image to latent
    /// 2. Add noise to the latent (controlled by strength)
    /// 3. Denoise with text guidance
    /// 4. Decode back to image
    ///
    /// <b>For Beginners:</b> This transforms an existing image based on a prompt:
    /// - strength=0.3: Minor changes, keeps most of the original
    /// - strength=0.7: Major changes, but composition remains
    /// - strength=1.0: Complete regeneration, original is just a starting point
    /// </para>
    /// </remarks>
    Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.8,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null);

    /// <summary>
    /// Performs inpainting (filling in masked regions).
    /// </summary>
    /// <param name="inputImage">The input image with areas to inpaint.</param>
    /// <param name="mask">Binary mask where 1 = inpaint, 0 = keep original.</param>
    /// <param name="prompt">Text prompt describing what to generate in the masked area.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The inpainted image tensor.</returns>
    /// <remarks>
    /// <para>
    /// Inpainting fills in masked regions while keeping unmasked areas intact.
    /// The mask should be the same spatial size as the image.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like a smart "fill" tool:
    /// - Draw a mask over what you want to replace
    /// - Describe what should go there
    /// - The model generates content that blends naturally
    /// </para>
    /// </remarks>
    Tensor<T> Inpaint(
        Tensor<T> inputImage,
        Tensor<T> mask,
        string prompt,
        string? negativePrompt = null,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null);

    /// <summary>
    /// Sets the guidance scale for classifier-free guidance.
    /// </summary>
    /// <param name="scale">The guidance scale (typically 1.0-20.0).</param>
    void SetGuidanceScale(double scale);

    /// <summary>
    /// Gets whether this model supports negative prompts.
    /// </summary>
    bool SupportsNegativePrompt { get; }

    /// <summary>
    /// Gets whether this model supports inpainting.
    /// </summary>
    bool SupportsInpainting { get; }
}
