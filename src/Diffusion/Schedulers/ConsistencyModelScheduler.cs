using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Consistency Model scheduler for single-step or few-step diffusion sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Consistency Models map any point on the probability flow ODE trajectory
/// directly to the trajectory's origin (the clean data). This allows single-step
/// generation, with optional multi-step refinement for higher quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> Consistency models are the fastest type of diffusion model:
///
/// Normal diffusion: Start with noise, take 20-50 small steps to get a clean image
/// Consistency model: Start with noise, jump directly to the clean image in 1 step!
///
/// Key characteristics:
/// - Single-step generation possible (fastest diffusion method)
/// - Multi-step mode (2-4 steps) improves quality
/// - Maps noisy samples directly to clean data predictions
/// - Works with both distilled and directly trained consistency models
///
/// How multi-step consistency works:
/// 1. Start with pure noise at sigma_max
/// 2. Apply consistency function → get approximate clean image
/// 3. Add noise back at a lower sigma level
/// 4. Apply consistency function again → better clean image
/// 5. Repeat for desired number of steps
///
/// This "denoise-then-add-noise" cycle progressively refines the output.
/// </para>
/// <para>
/// <b>Reference:</b> Song et al., "Consistency Models", ICML 2023
/// </para>
/// </remarks>
public sealed class ConsistencyModelScheduler<T> : NoiseSchedulerBase<T>
{
    private Vector<T>? _sigmas;
    private readonly Random _random;
    private readonly double _sigmaMin;
    private readonly double _sigmaMax;

    /// <summary>
    /// Initializes a new instance of the Consistency Model scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler.</param>
    /// <param name="sigmaMin">Minimum sigma (noise level). Default: 0.002.</param>
    /// <param name="sigmaMax">Maximum sigma (noise level). Default: 80.0.</param>
    /// <param name="seed">Optional random seed for multi-step noise injection.</param>
    public ConsistencyModelScheduler(
        SchedulerConfig<T> config,
        double sigmaMin = 0.002,
        double sigmaMax = 80.0,
        int? seed = null) : base(config)
    {
        _sigmaMin = sigmaMin;
        _sigmaMax = sigmaMax;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        ComputeSigmas(inferenceSteps);
    }

    /// <summary>
    /// Computes sigma schedule for consistency model multi-step sampling.
    /// Uses a geometrically-spaced sigma schedule from sigma_max to sigma_min.
    /// </summary>
    private void ComputeSigmas(int inferenceSteps)
    {
        _sigmas = new Vector<T>(inferenceSteps + 1);

        if (inferenceSteps == 1)
        {
            _sigmas[0] = NumOps.FromDouble(_sigmaMax);
            _sigmas[1] = NumOps.FromDouble(_sigmaMin);
            return;
        }

        // Geometrically-spaced sigmas from sigma_max to sigma_min
        double logSigmaMax = Math.Log(_sigmaMax);
        double logSigmaMin = Math.Log(_sigmaMin);

        for (int i = 0; i < inferenceSteps; i++)
        {
            double t = (double)i / (inferenceSteps - 1);
            double logSigma = logSigmaMax + t * (logSigmaMin - logSigmaMax);
            _sigmas[i] = NumOps.FromDouble(Math.Exp(logSigma));
        }

        _sigmas[inferenceSteps] = NumOps.FromDouble(_sigmaMin);
    }

    /// <summary>
    /// Performs one consistency model denoising step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The consistency model step:
    /// 1. Apply the consistency function to get x_0 prediction
    /// 2. If not the last step, add noise at the next sigma level
    /// 3. This creates the input for the next consistency function call
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

        // The consistency model output IS the predicted clean sample x_0
        // (consistency models are trained to directly output x_0)
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

        double sigmaNextDbl = NumOps.ToDouble(sigmaNext);

        // If this is the last step (sigma_next ≈ sigma_min), return the clean prediction
        if (sigmaNextDbl <= _sigmaMin + 1e-10)
        {
            return predOriginal;
        }

        // Multi-step: add noise at sigma_next level for the next iteration
        // x_next = pred_x0 + sigma_next * noise
        var stepNoise = noise ?? GenerateNoise(sample.Length);
        var noisedSample = Engine.Add(predOriginal, Engine.Multiply(stepNoise, sigmaNext));

        return noisedSample;
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
    /// Creates a Consistency Model scheduler with default settings.
    /// </summary>
    public static ConsistencyModelScheduler<T> CreateDefault(int? seed = null)
    {
        return new ConsistencyModelScheduler<T>(SchedulerConfig<T>.CreateDefault(), seed: seed);
    }
}
