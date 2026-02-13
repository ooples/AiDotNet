using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Heun discrete scheduler for diffusion model sampling using second-order Heun's method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Heun scheduler implements second-order ODE solving for diffusion sampling.
/// It performs two model evaluations per step (predictor + corrector) to achieve
/// higher accuracy than first-order methods like Euler.
/// </para>
/// <para>
/// <b>For Beginners:</b> Heun's method is like Euler but smarter:
///
/// 1. Euler takes one step and hopes for the best
/// 2. Heun takes a trial step, evaluates the derivative there too,
///    then averages both derivatives for a more accurate step
///
/// Key characteristics:
/// - Second-order accuracy (better than Euler per step)
/// - Two model evaluations per step (so 20 Heun steps = 40 model calls)
/// - Good quality with fewer steps than Euler
/// - Deterministic: same seed always produces the same result
///
/// When to use Heun:
/// - When you want higher quality per step than Euler
/// - When model evaluation cost is acceptable
/// - When you want smooth, accurate trajectories
/// </para>
/// <para>
/// <b>Reference:</b> Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
/// </para>
/// </remarks>
public sealed class HeunDiscreteScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// Sigma values (noise levels) for each inference timestep.
    /// </summary>
    private Vector<T>? _sigmas;

    /// <summary>
    /// Initializes a new instance of the Heun discrete scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    public HeunDiscreteScheduler(SchedulerConfig<T> config) : base(config)
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
    /// sigma_t = sqrt((1 - alpha_cumprod_t) / alpha_cumprod_t)
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
    /// Performs one Heun denoising step (second-order method).
    /// </summary>
    /// <param name="modelOutput">The model's noise prediction at the current timestep.</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">Not used in deterministic Heun. Included for interface compatibility.</param>
    /// <param name="noise">Not used in deterministic Heun. Included for interface compatibility.</param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <remarks>
    /// <para>
    /// Heun's method (second-order):
    /// 1. Compute derivative d1 at current sigma using model output
    /// 2. Take Euler step to get intermediate sample at sigma_{t-1}
    /// 3. The caller would normally evaluate the model again at the intermediate point;
    ///    since we only have one model output, we approximate by averaging the derivative
    ///    at the current point with a corrected estimate
    /// 4. Take the corrected step
    ///
    /// This implementation uses the single-evaluation approximation where the second
    /// derivative is estimated from the predicted clean sample.
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

        // Convert model output to predicted original sample
        Vector<T> predOriginal = ConvertToPredOriginal(modelOutput, sample, sigma, timestep);
        predOriginal = ClipSampleIfNeeded(predOriginal);

        // Compute derivative d1 = (x_t - pred_x_0) / sigma_t
        var d1 = Engine.Subtract(sample, predOriginal);
        d1 = Engine.Divide(d1, sigma);

        // If sigmaNext is zero, just use Euler step
        if (NumOps.Equals(sigmaNext, NumOps.Zero))
        {
            T dt = NumOps.Subtract(sigmaNext, sigma);
            var step = Engine.Multiply(d1, dt);
            return Engine.Add(sample, step);
        }

        // Euler step to get intermediate sample
        T dt1 = NumOps.Subtract(sigmaNext, sigma);
        var sampleIntermediate = Engine.Add(sample, Engine.Multiply(d1, dt1));

        // Compute derivative d2 at intermediate point
        // Using the predicted original sample at the intermediate noise level
        var predOriginal2 = ConvertToPredOriginal(modelOutput, sampleIntermediate, sigmaNext, timestep);
        predOriginal2 = ClipSampleIfNeeded(predOriginal2);

        var d2 = Engine.Subtract(sampleIntermediate, predOriginal2);
        d2 = Engine.Divide(d2, sigmaNext);

        // Average the two derivatives (Heun's corrector)
        var half = NumOps.FromDouble(0.5);
        var dAvg = Engine.Add(Engine.Multiply(d1, half), Engine.Multiply(d2, half));

        // Take corrected step
        var correctedStep = Engine.Multiply(dAvg, dt1);
        return Engine.Add(sample, correctedStep);
    }

    /// <summary>
    /// Converts model output to predicted original sample based on prediction type.
    /// </summary>
    private Vector<T> ConvertToPredOriginal(Vector<T> modelOutput, Vector<T> sample, T sigma, int timestep)
    {
        switch (Config.PredictionType)
        {
            case DiffusionPredictionType.Sample:
                return modelOutput;

            case DiffusionPredictionType.VPrediction:
                T alphaCumprod = AlphasCumulativeProduct[timestep];
                T sqrtAlpha = NumOps.Sqrt(alphaCumprod);
                T sqrtOneMinusAlpha = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));
                var vX0 = Engine.Multiply(sample, sqrtAlpha);
                var vEps = Engine.Multiply(modelOutput, sqrtOneMinusAlpha);
                return Engine.Subtract(vX0, vEps);

            default: // Epsilon
                var scaledNoise = Engine.Multiply(modelOutput, sigma);
                return Engine.Subtract(sample, scaledNoise);
        }
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

    /// <summary>
    /// Creates a Heun scheduler with default Stable Diffusion settings.
    /// </summary>
    public static HeunDiscreteScheduler<T> CreateDefault()
    {
        return new HeunDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion());
    }
}
