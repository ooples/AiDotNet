namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for noise prediction networks used in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Noise predictors are the core neural networks in diffusion models that learn to predict
/// the noise added to samples at each timestep. They can be implemented as U-Nets,
/// Diffusion Transformers (DiT), or other architectures.
/// </para>
/// <para>
/// <b>For Beginners:</b> A noise predictor is like a "noise detective" that looks at a noisy image
/// and figures out exactly what noise was added to it.
///
/// How it works:
/// 1. The model receives a noisy image and a timestep
/// 2. The timestep tells the model how much noise should be in the image
/// 3. The model predicts what noise pattern was added
/// 4. This prediction is used to remove noise and recover the original image
///
/// Different architectures for noise prediction:
/// - U-Net: The original and most common, uses an encoder-decoder with skip connections
/// - DiT (Diffusion Transformer): Uses transformer blocks, powers state-of-the-art models like SD3 and Sora
/// - U-ViT: Hybrid of U-Net and Vision Transformer
///
/// The architecture choice affects:
/// - Quality of generated images
/// - Speed of generation
/// - Memory requirements
/// - Ability to scale to larger models
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> to provide all standard
/// model capabilities (training, saving, loading, gradients, checkpointing, etc.).
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("NoisePredictor")]
public interface INoisePredictor<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the number of input channels the predictor expects.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For image models, this is typically:
    /// - 4 for latent diffusion models (VAE latent channels)
    /// - 3 for pixel-space RGB models
    /// - Higher for models with additional conditioning channels
    /// </para>
    /// </remarks>
    int InputChannels { get; }

    /// <summary>
    /// Gets the number of output channels the predictor produces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Usually matches InputChannels since we predict noise of the same shape as input.
    /// Some architectures may predict additional outputs like variance.
    /// </para>
    /// </remarks>
    int OutputChannels { get; }

    /// <summary>
    /// Gets the base channel count used in the network architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This determines the model capacity. Common values:
    /// - 320 for Stable Diffusion 1.x and 2.x
    /// - 384 for Stable Diffusion XL (base)
    /// - 1024 for large DiT models
    /// </para>
    /// </remarks>
    int BaseChannels { get; }

    /// <summary>
    /// Gets the dimension of the time/timestep embedding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The timestep is embedded into a high-dimensional vector before being
    /// injected into the network. Typical values: 256, 512, 1024.
    /// </para>
    /// </remarks>
    int TimeEmbeddingDim { get; }

    /// <summary>
    /// Predicts the noise in a noisy sample at a given timestep.
    /// </summary>
    /// <param name="noisySample">The noisy input sample [batch, channels, height, width].</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="conditioning">Optional conditioning tensor (e.g., text embeddings).</param>
    /// <returns>The predicted noise tensor with the same shape as noisySample.</returns>
    /// <remarks>
    /// <para>
    /// This is the main forward pass of the noise predictor. Given a noisy sample
    /// at timestep t, it predicts what noise was added.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where the actual denoising happens:
    /// 1. The network looks at the noisy image
    /// 2. It considers how noisy it should be at this timestep
    /// 3. It predicts the noise pattern
    /// 4. This prediction is subtracted to get a cleaner image
    /// </para>
    /// </remarks>
    Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null);

    /// <summary>
    /// Predicts noise with explicit timestep embedding (for batched different timesteps).
    /// </summary>
    /// <param name="noisySample">The noisy input sample [batch, channels, height, width].</param>
    /// <param name="timeEmbedding">Pre-computed timestep embeddings [batch, timeEmbeddingDim].</param>
    /// <param name="conditioning">Optional conditioning tensor (e.g., text embeddings).</param>
    /// <returns>The predicted noise tensor with the same shape as noisySample.</returns>
    /// <remarks>
    /// <para>
    /// This overload is useful when you want to use different timesteps per sample
    /// in a batch, or when you have pre-computed timestep embeddings for efficiency.
    /// </para>
    /// </remarks>
    Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null);

    /// <summary>
    /// Computes the timestep embedding for a given timestep.
    /// </summary>
    /// <param name="timestep">The timestep to embed.</param>
    /// <returns>The timestep embedding vector [timeEmbeddingDim].</returns>
    /// <remarks>
    /// <para>
    /// Timesteps are typically embedded using sinusoidal positional encodings
    /// (like in Transformers) followed by a small MLP.
    /// </para>
    /// </remarks>
    Tensor<T> GetTimestepEmbedding(int timestep);

    /// <summary>
    /// Gets whether this noise predictor supports classifier-free guidance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Classifier-free guidance allows steering generation toward the conditioning
    /// (e.g., text prompt) without a separate classifier. Most modern models support this.
    /// </para>
    /// </remarks>
    bool SupportsCFG { get; }

    /// <summary>
    /// Gets whether this noise predictor supports cross-attention conditioning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Cross-attention allows the model to attend to conditioning tokens (like text embeddings).
    /// This is how text-to-image models incorporate the prompt.
    /// </para>
    /// </remarks>
    bool SupportsCrossAttention { get; }

    /// <summary>
    /// Gets the expected context dimension for cross-attention conditioning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For CLIP-conditioned models, this is typically 768 or 1024.
    /// For T5-conditioned models (like SD3), this is typically 2048.
    /// Returns 0 if cross-attention is not supported.
    /// </para>
    /// </remarks>
    int ContextDimension { get; }
}
