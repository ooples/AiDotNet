using AiDotNet.Enums;

namespace AiDotNet.Diffusion;

/// <summary>
/// DPM++ 2M SDE scheduler — stochastic variant of DPM-Solver++ multistep.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DPM++ 2M SDE adds stochastic noise injection to the DPM-Solver++ 2M method.
/// This creates more diverse outputs while maintaining the efficiency of the
/// deterministic variant. The SDE formulation adds controlled randomness at each step.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the "SDE Karras" sampler popular in Stable Diffusion UIs:
///
/// - DPM++ 2M: Fast, deterministic, same seed = same image
/// - DPM++ 2M SDE: Adds slight randomness for more creative/diverse results
///
/// Key characteristics:
/// - Stochastic: Adds controlled noise at each step
/// - Two-step method with previous derivative memory
/// - Popular in community UIs (often labeled "DPM++ 2M SDE Karras")
/// - Good diversity-quality tradeoff
/// - Works well with 20-30 steps
///
/// The SDE noise strength is controlled by the eta parameter.
/// Higher eta = more stochastic = more diverse but potentially lower quality.
/// </para>
/// <para>
/// <b>Reference:</b> Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", NeurIPS 2022
/// </para>
/// </remarks>
public sealed class DPMSolverSDEScheduler<T> : NoiseSchedulerBase<T>
{
    private Vector<T>? _sigmas;
    private Vector<T>? _previousDerivative;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the DPM++ 2M SDE scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler.</param>
    /// <param name="seed">Optional random seed for reproducible stochastic sampling.</param>
    public DPMSolverSDEScheduler(SchedulerConfig<T> config, int? seed = null) : base(config)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        ComputeSigmas();
        _previousDerivative = null;
    }

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
    /// Performs one DPM++ 2M SDE denoising step with stochastic noise injection.
    /// </summary>
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

        // Compute derivative d = (x_t - pred_x_0) / sigma_t
        var derivative = Engine.Subtract(sample, predOriginal);
        derivative = Engine.Divide(derivative, sigma);

        // DPM++ 2M: use average of current and previous derivative if available
        Vector<T> effectiveDerivative;
        if (_previousDerivative != null)
        {
            var half = NumOps.FromDouble(0.5);
            effectiveDerivative = Engine.Add(
                Engine.Multiply(derivative, NumOps.FromDouble(1.5)),
                Engine.Multiply(_previousDerivative, NumOps.FromDouble(-0.5)));
        }
        else
        {
            effectiveDerivative = derivative;
        }

        _previousDerivative = derivative;

        // Deterministic step
        T dt = NumOps.Subtract(sigmaNext, sigma);
        var prevSample = Engine.Add(sample, Engine.Multiply(effectiveDerivative, dt));

        // SDE: Add stochastic noise proportional to eta and step size
        double etaValue = NumOps.ToDouble(eta);
        if (etaValue > 0.0 && NumOps.ToDouble(sigmaNext) > 0.0)
        {
            double sigmaD = NumOps.ToDouble(sigma);
            double sigmaNd = NumOps.ToDouble(sigmaNext);

            // Noise strength: sigma_noise = eta * sqrt(sigma_next^2 - sigma_down^2)
            // where sigma_down = sigma_next * sqrt(1 - eta^2)
            double sigmaDown = sigmaNd * Math.Sqrt(Math.Max(0.0, 1.0 - etaValue * etaValue));
            double noiseStrength = Math.Sqrt(Math.Max(0.0, sigmaNd * sigmaNd - sigmaDown * sigmaDown));

            if (noiseStrength > 1e-10)
            {
                // Generate or use provided noise
                var sdeNoise = noise ?? GenerateNoise(sample.Length);
                prevSample = Engine.Add(prevSample, Engine.Multiply(sdeNoise, NumOps.FromDouble(noiseStrength)));
            }
        }

        return prevSample;
    }

    /// <summary>
    /// Generates Gaussian noise with the scheduler's random state.
    /// </summary>
    private Vector<T> GenerateNoise(int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            // Box-Muller transform for Gaussian noise
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            result[i] = NumOps.FromDouble(normal);
        }

        return result;
    }

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
    /// Creates a DPM++ 2M SDE scheduler with default Stable Diffusion settings.
    /// </summary>
    public static DPMSolverSDEScheduler<T> CreateDefault(int? seed = null)
    {
        return new DPMSolverSDEScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion(), seed);
    }
}
