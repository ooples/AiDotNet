namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Interface for diffusion model step schedulers that control the noise schedule during inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> In diffusion models, a scheduler controls how noise is gradually
/// added to or removed from data. During training, noise is added step by step. During inference,
/// the scheduler reverses this process to generate clean samples from noise.
///
/// Key concepts:
/// - Timesteps: Discrete steps in the noise schedule (e.g., 1000 training steps, 50 inference steps)
/// - Beta schedule: Controls how much noise is added at each step
/// - Step function: Takes a noisy sample and model prediction, returns a slightly less noisy sample
/// </para>
/// </remarks>
public interface IStepScheduler<T>
{
    /// <summary>
    /// Gets the timesteps for the current inference schedule.
    /// </summary>
    int[] Timesteps { get; }

    /// <summary>
    /// Sets up the inference timesteps based on the number of steps desired.
    /// </summary>
    /// <param name="inferenceSteps">Number of denoising steps to use during inference.</param>
    void SetTimesteps(int inferenceSteps);

    /// <summary>
    /// Performs one denoising step using the model output.
    /// </summary>
    /// <param name="modelOutput">The model's prediction (typically noise prediction).</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">Stochasticity parameter (0 = deterministic DDIM, 1 = stochastic DDPM).</param>
    /// <param name="noise">Optional noise for stochastic sampling. If null and eta > 0, uses zeros (deterministic fallback).</param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null);
}
