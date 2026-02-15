using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Base class for diffusion model noise schedulers providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class implements the common behavior for all noise schedulers,
/// including beta schedule computation, alpha cumulative product calculation, noise addition,
/// and state management for checkpointing.
/// </para>
/// <para>
/// <b>Note:</b> This class was renamed from StepSchedulerBase to NoiseSchedulerBase to avoid
/// confusion with learning rate schedulers. Noise schedulers are specific to diffusion models.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation that all noise schedulers build upon.
/// It handles the common math and state management that every scheduler needs:
/// - Computing the noise schedule (how much noise at each step)
/// - Tracking the current state for saving/loading
/// - Adding noise during training
///
/// Specific schedulers like DDIM, PNDM, and DPM-Solver extend this base to implement
/// their unique denoising strategies.
/// </para>
/// </remarks>
public abstract class NoiseSchedulerBase<T> : INoiseScheduler<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a helper that knows how to do math with your
    /// specific number type, whether that's float, double, or decimal.
    /// </para>
    /// </remarks>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the compute engine for GPU-accelerated vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the configuration options for the scheduler.
    /// </summary>
    /// <inheritdoc />
    public SchedulerConfig<T> Config { get; }

    /// <summary>
    /// Beta values (noise variance) at each training timestep.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Beta represents how much noise is added at each step. Higher beta = more noise.
    /// These values typically increase from a small start value to a larger end value.
    /// </para>
    /// </remarks>
    protected Vector<T> Betas;

    /// <summary>
    /// Alpha values (1 - beta) representing signal retention at each timestep.
    /// </summary>
    protected Vector<T> Alphas;

    /// <summary>
    /// Cumulative product of alphas representing total signal retention at each timestep.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the key value for diffusion: alpha_cumprod[t] tells you what fraction
    /// of the original signal remains at timestep t. At t=0 it's ~1, at t=T it's ~0.
    /// </para>
    /// </remarks>
    protected Vector<T> AlphasCumulativeProduct;

    /// <summary>
    /// The timesteps for the current inference schedule.
    /// </summary>
    private int[] _timesteps = Array.Empty<int>();

    /// <inheritdoc />
    public int[] Timesteps => _timesteps;

    /// <inheritdoc />
    public int TrainTimesteps => Config.TrainTimesteps;

    /// <summary>
    /// Initializes a new instance of the NoiseSchedulerBase class.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <exception cref="NotSupportedException">Thrown when an unsupported beta schedule is specified.</exception>
    protected NoiseSchedulerBase(SchedulerConfig<T> config)
    {
        Guard.NotNull(config);
        Config = config;

        int steps = config.TrainTimesteps;
        Betas = new Vector<T>(steps);
        Alphas = new Vector<T>(steps);
        AlphasCumulativeProduct = new Vector<T>(steps);

        InitializeBetaSchedule();
        ComputeAlphas();
    }

    /// <summary>
    /// Initializes the beta schedule based on the configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The beta schedule determines how noise variance changes across timesteps.
    /// Linear schedules work well for many applications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This sets up "how much noise to add at each step."
    /// Linear means the noise amount increases steadily from start to end.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">Thrown when an unsupported beta schedule type is specified.</exception>
    private void InitializeBetaSchedule()
    {
        int steps = Config.TrainTimesteps;

        switch (Config.BetaSchedule)
        {
            case BetaSchedule.Linear:
                InitializeLinearBetaSchedule(steps);
                break;
            case BetaSchedule.ScaledLinear:
                InitializeScaledLinearBetaSchedule(steps);
                break;
            case BetaSchedule.SquaredCosine:
                InitializeSquaredCosineBetaSchedule(steps);
                break;
            default:
                throw new NotSupportedException($"Beta schedule '{Config.BetaSchedule}' is not supported.");
        }
    }

    /// <summary>
    /// Initializes a linear beta schedule.
    /// </summary>
    /// <param name="steps">Number of timesteps.</param>
    /// <remarks>
    /// Linear interpolation: beta[i] = betaStart + (betaEnd - betaStart) * i / (steps - 1)
    /// </remarks>
    private void InitializeLinearBetaSchedule(int steps)
    {
        var delta = NumOps.Subtract(Config.BetaEnd, Config.BetaStart);
        var stepsMinusOne = NumOps.FromDouble(steps - 1);

        for (int i = 0; i < steps; i++)
        {
            var ratio = NumOps.Divide(NumOps.FromDouble(i), stepsMinusOne);
            Betas[i] = NumOps.Add(Config.BetaStart, NumOps.Multiply(delta, ratio));
        }
    }

    /// <summary>
    /// Initializes a scaled linear beta schedule (used by Stable Diffusion).
    /// </summary>
    /// <param name="steps">Number of timesteps.</param>
    /// <remarks>
    /// Scaled linear uses sqrt of linear interpolated values, common in image generation.
    /// </remarks>
    private void InitializeScaledLinearBetaSchedule(int steps)
    {
        var sqrtStart = NumOps.Sqrt(Config.BetaStart);
        var sqrtEnd = NumOps.Sqrt(Config.BetaEnd);
        var delta = NumOps.Subtract(sqrtEnd, sqrtStart);
        var stepsMinusOne = NumOps.FromDouble(steps - 1);

        for (int i = 0; i < steps; i++)
        {
            var ratio = NumOps.Divide(NumOps.FromDouble(i), stepsMinusOne);
            var sqrtBeta = NumOps.Add(sqrtStart, NumOps.Multiply(delta, ratio));
            Betas[i] = NumOps.Multiply(sqrtBeta, sqrtBeta);
        }
    }

    /// <summary>
    /// Initializes a squared cosine beta schedule (improved schedule).
    /// </summary>
    /// <param name="steps">Number of timesteps.</param>
    /// <remarks>
    /// Squared cosine provides smoother noise progression and often better results.
    /// Based on "Improved Denoising Diffusion Probabilistic Models" paper.
    /// </remarks>
    private void InitializeSquaredCosineBetaSchedule(int steps)
    {
        var s = NumOps.FromDouble(0.008); // Small offset to prevent beta from being exactly 0 or 1

        for (int i = 0; i < steps; i++)
        {
            var t1 = NumOps.FromDouble((double)i / steps);
            var t2 = NumOps.FromDouble((double)(i + 1) / steps);

            var cos1 = NumOps.FromDouble(Math.Cos((NumOps.ToDouble(NumOps.Add(t1, s)) / (1.0 + NumOps.ToDouble(s))) * Math.PI / 2));
            var cos2 = NumOps.FromDouble(Math.Cos((NumOps.ToDouble(NumOps.Add(t2, s)) / (1.0 + NumOps.ToDouble(s))) * Math.PI / 2));

            var alphaCumprod1 = NumOps.Multiply(cos1, cos1);
            var alphaCumprod2 = NumOps.Multiply(cos2, cos2);

            var beta = NumOps.Subtract(NumOps.One, NumOps.Divide(alphaCumprod2, alphaCumprod1));
            // Clip beta to [0, 0.999] for numerical stability
            var maxBeta = NumOps.FromDouble(0.999);
            Betas[i] = NumOps.LessThan(beta, NumOps.Zero) ? NumOps.Zero :
                      (NumOps.GreaterThan(beta, maxBeta) ? maxBeta : beta);
        }
    }

    /// <summary>
    /// Computes alpha values and their cumulative products from betas.
    /// </summary>
    private void ComputeAlphas()
    {
        int steps = Config.TrainTimesteps;
        var cumulativeProduct = NumOps.One;

        for (int i = 0; i < steps; i++)
        {
            Alphas[i] = NumOps.Subtract(NumOps.One, Betas[i]);
            cumulativeProduct = NumOps.Multiply(cumulativeProduct, Alphas[i]);
            AlphasCumulativeProduct[i] = cumulativeProduct;
        }
    }

    /// <inheritdoc />
    public virtual void SetTimesteps(int inferenceSteps)
    {
        if (inferenceSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(inferenceSteps), "Inference steps must be positive.");
        if (inferenceSteps > Config.TrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(inferenceSteps),
                $"Inference steps ({inferenceSteps}) cannot exceed training timesteps ({Config.TrainTimesteps}).");

        int totalSteps = Config.TrainTimesteps;
        int stride = totalSteps / inferenceSteps;
        if (stride < 1) stride = 1;

        var timestepList = new List<int>();
        for (int i = totalSteps - 1; i >= 0 && timestepList.Count < inferenceSteps; i -= stride)
        {
            timestepList.Add(i);
        }

        _timesteps = timestepList.ToArray();
    }

    /// <inheritdoc />
    public abstract Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null);

    /// <inheritdoc />
    public virtual T GetAlphaCumulativeProduct(int timestep)
    {
        if (timestep < 0 || timestep >= AlphasCumulativeProduct.Length)
            throw new ArgumentOutOfRangeException(nameof(timestep),
                $"Timestep must be between 0 and {AlphasCumulativeProduct.Length - 1}.");

        return AlphasCumulativeProduct[timestep];
    }

    /// <inheritdoc />
    public virtual Vector<T> AddNoise(Vector<T> originalSample, Vector<T> noise, int timestep)
    {
        if (originalSample == null)
            throw new ArgumentNullException(nameof(originalSample));
        if (noise == null)
            throw new ArgumentNullException(nameof(noise));
        if (originalSample.Length != noise.Length)
            throw new ArgumentException("Original sample and noise must have the same length.", nameof(noise));
        if (timestep < 0 || timestep >= AlphasCumulativeProduct.Length)
            throw new ArgumentOutOfRangeException(nameof(timestep));

        // Forward diffusion: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        var alphaCumprod = AlphasCumulativeProduct[timestep];
        var sqrtAlphaCumprod = NumOps.Sqrt(alphaCumprod);
        var sqrtOneMinusAlphaCumprod = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));

        // Vectorized operations using IEngine for GPU acceleration
        var signalPart = Engine.Multiply(originalSample, sqrtAlphaCumprod);
        var noisePart = Engine.Multiply(noise, sqrtOneMinusAlphaCumprod);
        return Engine.Add(signalPart, noisePart);
    }

    /// <inheritdoc />
    public virtual Dictionary<string, object> GetState()
    {
        return new Dictionary<string, object>
        {
            ["timesteps"] = _timesteps.ToArray(),
            ["train_timesteps"] = Config.TrainTimesteps,
            ["beta_start"] = Config.BetaStart!,
            ["beta_end"] = Config.BetaEnd!,
            ["beta_schedule"] = Config.BetaSchedule.ToString(),
            ["prediction_type"] = Config.PredictionType.ToString()
        };
    }

    /// <inheritdoc />
    public virtual void LoadState(Dictionary<string, object> state)
    {
        if (state.TryGetValue("timesteps", out var timestepsObj) && timestepsObj is int[] timesteps)
        {
            _timesteps = timesteps;
        }
    }

    /// <summary>
    /// Validates common step parameters.
    /// </summary>
    /// <param name="modelOutput">The model output to validate.</param>
    /// <param name="sample">The sample to validate.</param>
    /// <param name="timestep">The timestep to validate.</param>
    /// <exception cref="ArgumentNullException">Thrown when modelOutput or sample is null.</exception>
    /// <exception cref="ArgumentException">Thrown when lengths don't match.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when timestep is out of range.</exception>
    protected void ValidateStepParameters(Vector<T> modelOutput, Vector<T> sample, int timestep)
    {
        if (modelOutput == null)
            throw new ArgumentNullException(nameof(modelOutput));
        if (sample == null)
            throw new ArgumentNullException(nameof(sample));
        if (modelOutput.Length != sample.Length)
            throw new ArgumentException("Model output and sample must have the same length.", nameof(modelOutput));
        if (timestep < 0 || timestep >= AlphasCumulativeProduct.Length)
            throw new ArgumentOutOfRangeException(nameof(timestep),
                $"Timestep must be between 0 and {AlphasCumulativeProduct.Length - 1}.");
    }

    /// <summary>
    /// Clips sample values to [-1, 1] if configured.
    /// </summary>
    /// <param name="sample">The sample to potentially clip.</param>
    /// <returns>The clipped sample if ClipSample is true, otherwise the original sample.</returns>
    protected Vector<T> ClipSampleIfNeeded(Vector<T> sample)
    {
        if (!Config.ClipSample)
            return sample;

        // Vectorized clamp using IEngine for GPU acceleration
        var negOne = NumOps.Negate(NumOps.One);
        return Engine.Clamp(sample, negOne, NumOps.One);
    }
}
