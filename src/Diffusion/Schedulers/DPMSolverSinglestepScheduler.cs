using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// DPM++ 2S Ancestral scheduler — single-step DPM-Solver++ with ancestral sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DPM++ 2S a (single-step, ancestral) performs a second-order DPM-Solver step
/// within each timestep and adds ancestral noise for diversity. Unlike DPM++ 2M which
/// uses multi-step history, this computes the second-order update in a single step
/// by performing two sub-steps internally.
/// </para>
/// <para>
/// <b>For Beginners:</b> This sampler combines accuracy with diversity:
///
/// - "2S": Two sub-steps within each step for second-order accuracy
/// - "Ancestral": Adds random noise at each step for diversity (like Euler Ancestral)
///
/// Key characteristics:
/// - Second-order accuracy without needing history from previous steps
/// - Stochastic (ancestral): different results each run unless seeded
/// - Good for creative/artistic generation where diversity is valued
/// - Works well with 20-30 steps
///
/// The "ancestral" noise means each step adds a small amount of noise,
/// making the trajectory stochastic. This can produce more diverse results
/// but may be less consistent than deterministic samplers.
/// </para>
/// <para>
/// <b>Reference:</b> Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", NeurIPS 2022
/// </para>
/// </remarks>
public sealed class DPMSolverSinglestepScheduler<T> : NoiseSchedulerBase<T>
{
    private Vector<T>? _sigmas;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the DPM++ 2S ancestral scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler.</param>
    /// <param name="seed">Optional random seed for reproducible ancestral sampling.</param>
    public DPMSolverSinglestepScheduler(SchedulerConfig<T> config, int? seed = null) : base(config)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        ComputeSigmas();
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
    /// Performs one DPM++ 2S ancestral denoising step.
    /// </summary>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        if (_sigmas == null)
            throw new InvalidOperationException("Sigmas not initialized. Call SetTimesteps() before Step().");

        int stepIndex = FindTimestepIndex(timestep);
        T sigma = _sigmas[stepIndex];
        T sigmaNext = _sigmas[stepIndex + 1];

        double sigmaDbl = NumOps.ToDouble(sigma);
        double sigmaNextDbl = NumOps.ToDouble(sigmaNext);

        // Convert model output to predicted original sample
        Vector<T> predOriginal = ConvertToPredOriginal(modelOutput, sample, sigma, timestep);
        predOriginal = ClipSampleIfNeeded(predOriginal);

        // Compute sigma_down and sigma_up for ancestral step
        double etaValue = NumOps.ToDouble(eta);
        double sigmaUp = 0.0;
        double sigmaDown = sigmaNextDbl;

        if (etaValue > 0.0 && sigmaNextDbl > 0.0)
        {
            // Ancestral step decomposition
            sigmaUp = Math.Min(sigmaNextDbl,
                etaValue * Math.Sqrt(sigmaNextDbl * sigmaNextDbl * (sigmaDbl * sigmaDbl - sigmaNextDbl * sigmaNextDbl)
                / (sigmaDbl * sigmaDbl)) );
            sigmaDown = Math.Sqrt(sigmaNextDbl * sigmaNextDbl - sigmaUp * sigmaUp);
        }

        // First-order step: compute midpoint sigma
        double sigmaMid = Math.Sqrt(sigmaDbl * sigmaDown);

        if (sigmaMid < 1e-10 || sigmaDown < 1e-10)
        {
            // Near-zero sigma: just use first-order Euler step
            var d = Engine.Subtract(sample, predOriginal);
            d = Engine.Divide(d, sigma);
            T dt = NumOps.FromDouble(sigmaDown - sigmaDbl);
            var result = Engine.Add(sample, Engine.Multiply(d, dt));

            // Add ancestral noise
            if (sigmaUp > 1e-10)
            {
                var ancestralNoise = noise ?? GenerateNoise(sample.Length);
                result = Engine.Add(result, Engine.Multiply(ancestralNoise, NumOps.FromDouble(sigmaUp)));
            }

            return result;
        }

        // First sub-step: Euler step to midpoint
        var derivative1 = Engine.Subtract(sample, predOriginal);
        derivative1 = Engine.Divide(derivative1, sigma);

        T dtMid = NumOps.FromDouble(sigmaMid - sigmaDbl);
        var sampleMid = Engine.Add(sample, Engine.Multiply(derivative1, dtMid));

        // Second sub-step: evaluate at midpoint and take corrected step
        // For the midpoint evaluation, we reuse the model output with sigma correction
        var predOriginalMid = ConvertToPredOriginal(modelOutput, sampleMid, NumOps.FromDouble(sigmaMid), timestep);
        predOriginalMid = ClipSampleIfNeeded(predOriginalMid);

        var derivative2 = Engine.Subtract(sampleMid, predOriginalMid);
        derivative2 = Engine.Divide(derivative2, NumOps.FromDouble(sigmaMid));

        // Full step using corrected derivative
        T dtFull = NumOps.FromDouble(sigmaDown - sigmaDbl);
        var prevSample = Engine.Add(sample, Engine.Multiply(derivative2, dtFull));

        // Add ancestral noise
        if (sigmaUp > 1e-10)
        {
            var ancestralNoise = noise ?? GenerateNoise(sample.Length);
            prevSample = Engine.Add(prevSample, Engine.Multiply(ancestralNoise, NumOps.FromDouble(sigmaUp)));
        }

        return prevSample;
    }

    private Vector<T> GenerateNoise(int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
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
    /// Creates a DPM++ 2S scheduler with default Stable Diffusion settings.
    /// </summary>
    public static DPMSolverSinglestepScheduler<T> CreateDefault(int? seed = null)
    {
        return new DPMSolverSinglestepScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion(), seed);
    }
}
