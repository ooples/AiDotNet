using AiDotNet.Enums;

namespace AiDotNet.NeuralNetworks.Diffusion.Schedulers;

/// <summary>
/// Euler Ancestral discrete scheduler for diffusion model sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Euler Ancestral scheduler combines Euler's method with ancestral sampling,
/// adding stochastic noise at each step. This creates more diverse outputs compared
/// to the deterministic Euler scheduler, at the cost of slightly less consistency.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the "Euler a" sampler commonly seen in Stable Diffusion UIs.
///
/// The difference from regular Euler:
/// - Euler: Deterministic - same seed always gives identical results
/// - Euler Ancestral: Stochastic - adds controlled randomness at each step
///
/// This stochasticity means:
/// - More creative/diverse outputs for the same prompt
/// - Slightly less reproducible (even with same seed, small changes cascade)
/// - Often produces more detailed, painterly results
/// - Popular choice for artistic/creative generation
///
/// The "ancestral" part means it samples from the reverse diffusion posterior,
/// similar to how DDPM adds noise at each step, but using Euler integration
/// for the deterministic part.
/// </para>
/// <para>
/// <b>Reference:</b> Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
/// </para>
/// </remarks>
public sealed class EulerAncestralDiscreteScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// Sigma values (noise levels) for each inference timestep.
    /// </summary>
    private Vector<T>? _sigmas;

    /// <summary>
    /// Initializes a new instance of the Euler Ancestral discrete scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create an Euler Ancestral scheduler for stochastic sampling.
    /// Works best with 20-50 inference steps.
    /// </para>
    /// <example>
    /// <code>
    /// var config = SchedulerConfig&lt;double&gt;.CreateStableDiffusion();
    /// var scheduler = new EulerAncestralDiscreteScheduler&lt;double&gt;(config);
    /// scheduler.SetTimesteps(30);
    /// </code>
    /// </example>
    /// </remarks>
    public EulerAncestralDiscreteScheduler(SchedulerConfig<T> config) : base(config)
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
    private void ComputeSigmas()
    {
        var timesteps = Timesteps;
        _sigmas = new Vector<T>(timesteps.Length + 1);

        for (int i = 0; i < timesteps.Length; i++)
        {
            T alphaCumprod = AlphasCumulativeProduct[timesteps[i]];
            T oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaCumprod);
            _sigmas[i] = NumOps.Sqrt(NumOps.Divide(oneMinusAlpha, alphaCumprod));
        }

        _sigmas[timesteps.Length] = NumOps.Zero;
    }

    /// <summary>
    /// Performs one Euler Ancestral denoising step.
    /// </summary>
    /// <param name="modelOutput">The model's noise prediction.</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">
    /// Controls the amount of ancestral noise:
    /// - 0 = fully deterministic (equivalent to regular Euler)
    /// - 1 = full ancestral sampling (default behavior)
    /// </param>
    /// <param name="noise">
    /// Random noise for ancestral sampling. Required for stochastic behavior.
    /// If null, falls back to deterministic Euler step.
    /// </param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <remarks>
    /// <para>
    /// Euler Ancestral step:
    /// 1. Compute predicted original sample from model output
    /// 2. Compute derivative d = (x - pred_x0) / sigma
    /// 3. Compute sigma_down (deterministic part) and sigma_up (stochastic part)
    /// 4. Euler step: x = x + d * (sigma_down - sigma)
    /// 5. Add ancestral noise: x = x + noise * sigma_up
    /// </para>
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        if (_sigmas == null)
            throw new InvalidOperationException("Sigmas not initialized. Call SetTimesteps() before Step().");

        int stepIndex = FindTimestepIndex(timestep);
        T sigma = _sigmas[stepIndex];
        T sigmaNext = _sigmas[stepIndex + 1];

        // Compute predicted original sample based on prediction type
        Vector<T> predOriginal;
        switch (Config.PredictionType)
        {
            case DiffusionPredictionType.Sample:
                predOriginal = modelOutput;
                break;

            case DiffusionPredictionType.VPrediction:
                T alphaCumprod = AlphasCumulativeProduct[timestep];
                T sqrtAlpha = NumOps.Sqrt(alphaCumprod);
                T sqrtOneMinusAlpha = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));
                var vX0 = Engine.Multiply(sample, sqrtAlpha);
                var vEps = Engine.Multiply(modelOutput, sqrtOneMinusAlpha);
                predOriginal = Engine.Subtract(vX0, vEps);
                break;

            default: // Epsilon
                var scaledNoise = Engine.Multiply(modelOutput, sigma);
                predOriginal = Engine.Subtract(sample, scaledNoise);
                break;
        }

        predOriginal = ClipSampleIfNeeded(predOriginal);

        // Compute derivative: d = (x - pred_x0) / sigma
        var derivative = Engine.Subtract(sample, predOriginal);
        derivative = Engine.Divide(derivative, sigma);

        // Compute sigma_down and sigma_up for ancestral sampling
        // sigma_up = eta * sqrt(sigma_next^2 * (sigma^2 - sigma_next^2) / sigma^2)
        // sigma_down = sqrt(sigma_next^2 - sigma_up^2)
        T sigmaUp = NumOps.Zero;
        T sigmaDown = sigmaNext;

        if (NumOps.GreaterThan(eta, NumOps.Zero) && NumOps.GreaterThan(sigmaNext, NumOps.Zero))
        {
            T sigmaSq = NumOps.Multiply(sigma, sigma);
            T sigmaNextSq = NumOps.Multiply(sigmaNext, sigmaNext);

            // sigma_up^2 = sigma_next^2 * (sigma^2 - sigma_next^2) / sigma^2
            T numerator = NumOps.Multiply(sigmaNextSq, NumOps.Subtract(sigmaSq, sigmaNextSq));
            T sigmaUpSq = NumOps.Divide(numerator, sigmaSq);

            // Ensure non-negative
            if (NumOps.GreaterThan(sigmaUpSq, NumOps.Zero))
            {
                sigmaUp = NumOps.Multiply(eta, NumOps.Sqrt(sigmaUpSq));
                T sigmaUpActualSq = NumOps.Multiply(sigmaUp, sigmaUp);
                T sigmaDownSq = NumOps.Subtract(sigmaNextSq, sigmaUpActualSq);
                sigmaDown = NumOps.GreaterThan(sigmaDownSq, NumOps.Zero)
                    ? NumOps.Sqrt(sigmaDownSq)
                    : NumOps.Zero;
            }
        }

        // Euler step with sigma_down
        T dt = NumOps.Subtract(sigmaDown, sigma);
        var step = Engine.Multiply(derivative, dt);
        var prevSample = Engine.Add(sample, step);

        // Add ancestral noise
        if (NumOps.GreaterThan(sigmaUp, NumOps.Zero) && noise != null)
        {
            var noiseTerm = Engine.Multiply(noise, sigmaUp);
            prevSample = Engine.Add(prevSample, noiseTerm);
        }

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
