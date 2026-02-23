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

    // Two-pass Heun state: stored between predictor (first call) and corrector (second call)
    private Vector<T>? _prevDerivative;
    private Vector<T>? _prevSample;
    private T _prevDt;
    private T _prevSigmaNext;
    private int _prevTimestep;
    private bool _isSecondPass;

    /// <summary>
    /// Initializes a new instance of the Heun discrete scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    public HeunDiscreteScheduler(SchedulerConfig<T> config) : base(config)
    {
        _prevDt = NumOps.Zero;
        _prevSigmaNext = NumOps.Zero;
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
    /// Heun's method requires two model evaluations per denoising step:
    ///
    /// Pass 1 (Predictor): Computes derivative d1 at current sigma, takes Euler step
    /// to intermediate point, returns intermediate sample for a second model evaluation.
    ///
    /// Pass 2 (Corrector): Uses the second model output at the intermediate point to
    /// compute d2, averages d1 and d2, and takes the corrected step.
    ///
    /// The caller must call Step() twice per denoising step:
    /// 1. intermediate = scheduler.Step(model(x_t), t, x_t, eta)
    /// 2. result = scheduler.Step(model(intermediate), t, intermediate, eta)
    /// </para>
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        if (_sigmas == null)
            throw new InvalidOperationException("Sigmas not initialized. Call SetTimesteps() before Step().");

        if (_isSecondPass && _prevDerivative is not null && _prevSample is not null)
        {
            if (timestep != _prevTimestep)
            {
                throw new InvalidOperationException(
                    $"Heun corrector step expected timestep {_prevTimestep} (matching the predictor), " +
                    $"but received {timestep}. Ensure the second Step() call uses the same timestep.");
            }

            return HeunCorrectorStep(modelOutput, sample);
        }

        return HeunPredictorStep(modelOutput, timestep, sample);
    }

    /// <summary>
    /// First pass: compute derivative d1, take Euler step, store state for corrector.
    /// </summary>
    private Vector<T> HeunPredictorStep(Vector<T> modelOutput, int timestep, Vector<T> sample)
    {
        int stepIndex = FindTimestepIndex(timestep);
        T sigma = _sigmas![stepIndex];
        T sigmaNext = _sigmas[stepIndex + 1];

        // Convert model output to predicted original sample
        Vector<T> predOriginal = ConvertToPredOriginal(modelOutput, sample, sigma, timestep);
        predOriginal = ClipSampleIfNeeded(predOriginal);

        // Compute derivative d1 = (x_t - pred_x_0) / sigma_t
        var d1 = Engine.Subtract(sample, predOriginal);
        d1 = Engine.Divide(d1, sigma);

        T dt = NumOps.Subtract(sigmaNext, sigma);

        // If sigmaNext is zero, just use Euler step (no corrector needed)
        if (NumOps.Equals(sigmaNext, NumOps.Zero))
        {
            _isSecondPass = false;
            var step = Engine.Multiply(d1, dt);
            return Engine.Add(sample, step);
        }

        // Euler step to get intermediate sample
        var sampleIntermediate = Engine.Add(sample, Engine.Multiply(d1, dt));

        // Store state for corrector pass
        _prevDerivative = d1;
        _prevSample = sample;
        _prevDt = dt;
        _prevSigmaNext = sigmaNext;
        _prevTimestep = timestep;
        _isSecondPass = true;

        return sampleIntermediate;
    }

    /// <summary>
    /// Second pass: use the second model evaluation at the intermediate point
    /// to compute the Heun corrector step.
    /// </summary>
    private Vector<T> HeunCorrectorStep(Vector<T> modelOutput, Vector<T> sampleIntermediate)
    {
        var d1 = _prevDerivative!;
        var originalSample = _prevSample!;
        T dt = _prevDt;
        T sigmaNext = _prevSigmaNext;
        int timestep = _prevTimestep;

        // Convert the second model output to predicted original sample at sigmaNext
        Vector<T> predOriginal2 = ConvertToPredOriginal(modelOutput, sampleIntermediate, sigmaNext, timestep);
        predOriginal2 = ClipSampleIfNeeded(predOriginal2);

        // Compute derivative d2 = (x_intermediate - pred_x0_2) / sigma_next
        var d2 = Engine.Subtract(sampleIntermediate, predOriginal2);
        d2 = Engine.Divide(d2, sigmaNext);

        // Clear state
        _isSecondPass = false;
        _prevDerivative = null;
        _prevSample = null;

        // Average the two derivatives (Heun's corrector)
        var half = NumOps.FromDouble(0.5);
        var dAvg = Engine.Add(Engine.Multiply(d1, half), Engine.Multiply(d2, half));

        // Take corrected step from the ORIGINAL sample
        var correctedStep = Engine.Multiply(dAvg, dt);
        return Engine.Add(originalSample, correctedStep);
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
