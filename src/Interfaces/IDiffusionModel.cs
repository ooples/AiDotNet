namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for diffusion-based generative models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Diffusion models are a class of generative models that learn to create data by reversing
/// a gradual noising process. They have achieved state-of-the-art results in image generation,
/// audio synthesis, and other generative tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Diffusion models are like learning to reverse a process of adding static to a TV signal.
///
/// How diffusion works:
/// 1. Forward process (training): Start with real data, gradually add noise until it's pure static
/// 2. Reverse process (generation): Start with pure static, gradually remove noise to create new data
///
/// The model learns: "Given this noisy version, what did the original look like?"
///
/// This is different from other generative models:
/// - GANs: Two networks competing (generator vs discriminator)
/// - VAEs: Compress and decompress through a bottleneck
/// - Diffusion: Iteratively denoise from random noise
///
/// Diffusion models are known for:
/// - High quality outputs (often better than GANs)
/// - Stable training (no mode collapse)
/// - Good diversity (produces varied outputs)
/// - Slower generation (many denoising steps needed)
/// </para>
/// <para>
/// <b>Key components:</b>
/// - Noise prediction model: A neural network that predicts noise in images
/// - Noise scheduler: Controls the noise schedule (see <see cref="INoiseScheduler{T}"/>)
/// - Loss function: Measures how well the model predicts noise (usually MSE)
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> to provide a consistent API
/// for diffusion models while inheriting all the standard model capabilities (training, saving,
/// loading, gradients, checkpointing, etc.).
/// </para>
/// </remarks>
public interface IDiffusionModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the step scheduler used for the diffusion process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The scheduler controls the noise schedule and denoising steps during generation.
    /// Different schedulers offer different tradeoffs between quality and speed:
    /// - DDPM: Original scheduler, high quality but slow (1000 steps)
    /// - DDIM: Deterministic, allows fewer steps (20-100)
    /// - PNDM: Fast multi-step scheduler (20-50 steps)
    /// </para>
    /// </remarks>
    INoiseScheduler<T> Scheduler { get; }

    /// <summary>
    /// Generates samples by iteratively denoising from random noise.
    /// </summary>
    /// <param name="shape">The shape of samples to generate (e.g., [batchSize, channels, height, width]).</param>
    /// <param name="numInferenceSteps">Number of denoising steps. More steps = higher quality, slower.</param>
    /// <param name="seed">Optional random seed for reproducibility. If null, uses system random.</param>
    /// <returns>Generated samples as a tensor.</returns>
    /// <remarks>
    /// <para>
    /// This is the main generation method. It starts with random noise and applies
    /// the reverse diffusion process to generate new samples.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how you create new images/data:
    /// 1. Start with pure random noise (like TV static)
    /// 2. Ask the model "what does this look like minus some noise?"
    /// 3. Repeat many times, each time removing a bit more noise
    /// 4. End with a clean generated sample
    ///
    /// More inference steps = cleaner results but slower generation.
    /// Typical values: 20-50 for fast generation, 100-200 for high quality.
    /// </para>
    /// </remarks>
    Tensor<T> Generate(int[] shape, int numInferenceSteps = 50, int? seed = null);

    /// <summary>
    /// Predicts the noise in a noisy sample at a given timestep.
    /// </summary>
    /// <param name="noisySample">The noisy input sample.</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <returns>The predicted noise tensor.</returns>
    /// <remarks>
    /// <para>
    /// This is the core prediction that the model learns. Given a noisy sample at
    /// timestep t, predict what noise was added to create it.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The model looks at a noisy image and guesses
    /// "what noise was added to make it look like this?" This prediction is then
    /// used to remove that noise and get a cleaner image.
    /// </para>
    /// </remarks>
    Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep);

    /// <summary>
    /// Computes the training loss for a batch of samples.
    /// </summary>
    /// <param name="cleanSamples">The original clean samples.</param>
    /// <param name="noise">The noise to add (typically sampled from standard normal).</param>
    /// <param name="timesteps">The timesteps at which to compute loss (one per sample).</param>
    /// <returns>The computed loss value.</returns>
    /// <remarks>
    /// <para>
    /// The standard diffusion training loss is the mean squared error between
    /// the actual noise and the model's predicted noise.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> During training:
    /// 1. Take a clean image
    /// 2. Add known noise to it at a random timestep
    /// 3. Ask the model to predict what noise was added
    /// 4. Compare the prediction to the actual noise (this is the loss)
    /// 5. Update the model to make better predictions
    /// </para>
    /// </remarks>
    T ComputeLoss(Tensor<T> cleanSamples, Tensor<T> noise, int[] timesteps);
}
