using AiDotNet.Enums;

namespace AiDotNet.Diffusion;

/// <summary>
/// Euler discrete scheduler for diffusion model sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Euler scheduler implements first-order ODE solving for diffusion sampling.
/// It uses Euler's method to solve the probability flow ODE, providing fast
/// deterministic sampling with good quality at moderate step counts (20-50 steps).
/// </para>
/// <para>
/// <b>For Beginners:</b> The Euler scheduler is one of the most popular sampling methods.
///
/// Imagine you're navigating from point A (pure noise) to point B (clean image):
/// - DDPM: Takes tiny, cautious steps with some randomness
/// - Euler: Uses calculus (ODE solving) to take direct, efficient steps
///
/// Key characteristics:
/// - Deterministic: Same seed always produces the same result
/// - Fast convergence: Good results in 20-50 steps
/// - Simple and efficient: Low computational overhead per step
/// - Widely used in Stable Diffusion UIs (often called "Euler" or "Euler a")
///
/// The Euler method converts sigma (noise level) at each step and predicts
/// the "derivative" of the denoising trajectory, then takes a step along it.
/// </para>
/// <para>
/// <b>Reference:</b> Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
/// </para>
/// </remarks>
public sealed class EulerDiscreteScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// Sigma values (noise levels) for each inference timestep.
    /// </summary>
    private Vector<T>? _sigmas;

    /// <summary>
    /// Initializes a new instance of the Euler discrete scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create an Euler scheduler for efficient deterministic sampling.
    /// Works best with 20-50 inference steps.
    /// </para>
    /// <example>
    /// <code>
    /// var config = SchedulerConfig&lt;double&gt;.CreateStableDiffusion();
    /// var scheduler = new EulerDiscreteScheduler&lt;double&gt;(config);
    /// scheduler.SetTimesteps(30);
    /// </code>
    /// </example>
    /// </remarks>
    public EulerDiscreteScheduler(SchedulerConfig<T> config) : base(config)
    {
    }

    /// <summary>
    /// Sets up the inference timesteps and computes sigma schedule.
    /// </summary>
    /// <param name="inferenceSteps">Number of denoising steps to use during inference.</param>
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        ComputeSigmas();
    }

    /// <summary>
    /// Computes sigma values from the alpha cumulative product schedule.
    /// </summary>
    /// <remarks>
    /// sigma_t = sqrt((1 - alpha_cumprod_t) / alpha_cumprod_t)
    /// This converts the alpha schedule to a noise level schedule used by Euler methods.
    /// </remarks>
    private void ComputeSigmas()
    {
        var timesteps = Timesteps;
        _sigmas = new Vector<T>(timesteps.Length + 1);

        for (int i = 0; i < timesteps.Length; i++)
        {
            T alphaCumprod = AlphasCumulativeProduct[timesteps[i]];
            T oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaCumprod);
            // sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
            _sigmas[i] = NumOps.Sqrt(NumOps.Divide(oneMinusAlpha, alphaCumprod));
        }

        // Append sigma=0 for the final (clean) state
        _sigmas[timesteps.Length] = NumOps.Zero;
    }

    /// <summary>
    /// Performs one Euler discrete denoising step.
    /// </summary>
    /// <param name="modelOutput">The model's noise prediction.</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">Not used in deterministic Euler. Included for interface compatibility.</param>
    /// <param name="noise">Not used in deterministic Euler. Included for interface compatibility.</param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <remarks>
    /// <para>
    /// Euler discrete step:
    /// 1. Convert model output to "derivative" (d) of the ODE
    /// 2. Compute step size: dt = sigma_{t-1} - sigma_t
    /// 3. Take Euler step: x_{t-1} = x_t + d * dt
    /// </para>
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        if (_sigmas == null)
            throw new InvalidOperationException("Sigmas not initialized. Call SetTimesteps() before Step().");

        // Find the index of the current timestep in the schedule
        int stepIndex = FindTimestepIndex(timestep);
        T sigma = _sigmas[stepIndex];
        T sigmaNext = _sigmas[stepIndex + 1];

        // Convert model output to denoised prediction based on prediction type
        Vector<T> predOriginal;
        switch (Config.PredictionType)
        {
            case DiffusionPredictionType.Sample:
                predOriginal = modelOutput;
                break;

            case DiffusionPredictionType.VPrediction:
                // For v-prediction: x_0 = sigma * sample - (sigma^2 + 1).sqrt() ...
                // Simplified: use alpha_cumprod conversion
                T alphaCumprod = AlphasCumulativeProduct[timestep];
                T sqrtAlpha = NumOps.Sqrt(alphaCumprod);
                T sqrtOneMinusAlpha = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));
                var vX0 = Engine.Multiply(sample, sqrtAlpha);
                var vEps = Engine.Multiply(modelOutput, sqrtOneMinusAlpha);
                predOriginal = Engine.Subtract(vX0, vEps);
                break;

            default: // Epsilon
                // x_0 = (sample - sigma * eps) / ... using the Karras formulation:
                // d = (sample - pred_original) / sigma
                // For epsilon prediction: pred_original = sample - sigma * eps
                var scaledNoise = Engine.Multiply(modelOutput, sigma);
                predOriginal = Engine.Subtract(sample, scaledNoise);
                break;
        }

        predOriginal = ClipSampleIfNeeded(predOriginal);

        // Compute the "derivative" d = (x_t - pred_x_0) / sigma_t
        var derivative = Engine.Subtract(sample, predOriginal);
        derivative = Engine.Divide(derivative, sigma);

        // Euler step: x_{t-1} = x_t + d * (sigma_{t-1} - sigma_t)
        T dt = NumOps.Subtract(sigmaNext, sigma);
        var step = Engine.Multiply(derivative, dt);
        var prevSample = Engine.Add(sample, step);

        return prevSample;
    }

    /// <summary>
    /// Finds the index of a timestep in the current schedule.
    /// </summary>
    private int FindTimestepIndex(int timestep)
    {
        var timesteps = Timesteps;
        for (int i = 0; i < timesteps.Length; i++)
        {
            if (timesteps[i] == timestep)
                return i;
        }

        // Fall back to closest timestep
        int closestIdx = 0;
        int closestDist = Math.Abs(timesteps[0] - timestep);
        for (int i = 1; i < timesteps.Length; i++)
        {
            int dist = Math.Abs(timesteps[i] - timestep);
            if (dist < closestDist)
            {
                closestDist = dist;
                closestIdx = i;
            }
        }

        return closestIdx;
    }
}
