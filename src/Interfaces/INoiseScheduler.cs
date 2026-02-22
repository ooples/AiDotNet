using AiDotNet.Diffusion;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for diffusion model noise schedulers that control the noise schedule during inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Noise schedulers are a core component of diffusion models that control how noise is gradually
/// added to or removed from data during the diffusion process. They define the noise schedule
/// (how much noise at each timestep) and provide the mathematical operations to denoise samples.
/// </para>
/// <para>
/// <b>Note:</b> This interface was renamed from IStepScheduler to INoiseScheduler to avoid
/// confusion with learning rate schedulers (ILearningRateScheduler). Noise schedulers are
/// specific to diffusion models, while learning rate schedulers control optimization dynamics.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a noise scheduler like a recipe for gradually revealing a hidden picture.
///
/// Imagine you have a clear photograph that you've covered with many layers of static (noise).
/// The scheduler tells you:
/// - How many layers of static there are (timesteps)
/// - How much static is in each layer (noise schedule)
/// - How to remove one layer at a time to gradually reveal the picture (step function)
///
/// Different schedulers (DDIM, PNDM, DPM-Solver) are like different techniques for removing
/// the static - some are faster, some produce better quality, and some offer a tradeoff.
///
/// Key concepts:
/// - Timesteps: Discrete steps in the noise schedule (e.g., 1000 training steps, 50 inference steps)
/// - Beta schedule: Controls how much noise is added at each step
/// - Step function: Takes a noisy sample and model prediction, returns a slightly less noisy sample
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("NoiseScheduler")]
public interface INoiseScheduler<T>
{
    /// <summary>
    /// Gets the timesteps for the current inference schedule.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the discrete time indices at which denoising steps will be performed.
    /// The array is typically in descending order (from highest noise to lowest).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like a list of checkpoints. If you have 50 inference
    /// steps, this array tells you exactly which of the original 1000 training timesteps
    /// to use for denoising.
    /// </para>
    /// </remarks>
    int[] Timesteps { get; }

    /// <summary>
    /// Gets the number of training timesteps this scheduler was configured with.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the total number of timesteps used during training, typically 1000.
    /// The scheduler interpolates between these for inference.
    /// </para>
    /// </remarks>
    int TrainTimesteps { get; }

    /// <summary>
    /// Sets up the inference timesteps based on the number of steps desired.
    /// </summary>
    /// <param name="inferenceSteps">Number of denoising steps to use during inference.</param>
    /// <remarks>
    /// <para>
    /// This method calculates which timesteps from the training schedule should be used
    /// for the given number of inference steps. Using fewer steps is faster but may
    /// reduce quality.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like choosing how many steps to take when walking
    /// from point A to point B. More steps (50-100) give smoother results, fewer steps
    /// (10-20) are faster but may miss details.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when inferenceSteps is less than 1 or greater than TrainTimesteps.
    /// </exception>
    void SetTimesteps(int inferenceSteps);

    /// <summary>
    /// Performs one denoising step using the model output.
    /// </summary>
    /// <param name="modelOutput">The model's prediction (typically noise prediction).</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">
    /// Stochasticity parameter: 0 = deterministic (DDIM), 1 = fully stochastic (DDPM).
    /// Values between 0 and 1 interpolate between these behaviors.
    /// </param>
    /// <param name="noise">
    /// Optional noise for stochastic sampling. If null and eta > 0, uses zero noise (deterministic fallback).
    /// </param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <remarks>
    /// <para>
    /// This is the core denoising operation. Given the current noisy sample and the model's
    /// prediction of what noise was added, it computes a slightly less noisy version.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is one step of "un-blurring" the image. The model
    /// looks at the current noisy image and guesses what noise is there. This method
    /// then removes that estimated noise to get a cleaner image.
    ///
    /// The eta parameter controls randomness:
    /// - eta=0: Always produces the same output for the same input (deterministic)
    /// - eta=1: Adds randomness, making each generation unique (stochastic)
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// Thrown when modelOutput or sample is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when modelOutput and sample have different lengths.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when timestep is negative or greater than TrainTimesteps.
    /// </exception>
    Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null);

    /// <summary>
    /// Gets the cumulative product of alphas (signal retention) at a given timestep.
    /// </summary>
    /// <param name="timestep">The timestep to query.</param>
    /// <returns>The cumulative alpha value at that timestep.</returns>
    /// <remarks>
    /// <para>
    /// Alpha cumulative product represents how much of the original signal is retained
    /// at each timestep. At t=0, it's close to 1 (mostly signal). At t=T, it's close
    /// to 0 (mostly noise).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you "how clear" the image is at each step.
    /// At the start (t=0), the image is clear (alpha near 1). At the end (t=T),
    /// it's pure noise (alpha near 0).
    /// </para>
    /// </remarks>
    T GetAlphaCumulativeProduct(int timestep);

    /// <summary>
    /// Adds noise to a clean sample according to the noise schedule.
    /// </summary>
    /// <param name="originalSample">The clean sample to add noise to.</param>
    /// <param name="noise">The noise to add.</param>
    /// <param name="timestep">The timestep determining how much noise to add.</param>
    /// <returns>The noisy sample.</returns>
    /// <remarks>
    /// <para>
    /// This implements the forward diffusion process: q(x_t | x_0) = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like adding a specific amount of static to a clear image.
    /// Higher timesteps add more noise. This is used during training to create noisy samples
    /// for the model to learn from.
    /// </para>
    /// </remarks>
    Vector<T> AddNoise(Vector<T> originalSample, Vector<T> noise, int timestep);

    /// <summary>
    /// Gets the scheduler configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The configuration contains the parameters used to create and initialize the scheduler,
    /// such as the beta schedule, prediction type, and training timesteps.
    /// </para>
    /// </remarks>
    SchedulerConfig<T> Config { get; }

    /// <summary>
    /// Gets the current scheduler state for checkpointing.
    /// </summary>
    /// <returns>A dictionary containing the scheduler's state.</returns>
    Dictionary<string, object> GetState();

    /// <summary>
    /// Loads scheduler state from a checkpoint.
    /// </summary>
    /// <param name="state">The state dictionary to load from.</param>
    void LoadState(Dictionary<string, object> state);
}
