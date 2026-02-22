using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion;

/// <summary>
/// Flow matching scheduler implementing rectified flow ODE sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Flow matching is a fundamentally different approach from DDPM-style diffusion. Instead of
/// learning to predict noise (epsilon) at each step, the model learns a velocity field v(x_t, t)
/// that defines an ordinary differential equation (ODE) transporting data between noise and signal.
/// </para>
/// <para>
/// <b>For Beginners:</b> Flow matching works like drawing a straight line from noise to image.
///
/// Traditional diffusion (DDPM/DDIM):
/// - Adds noise using: x_t = sqrt(alpha) * image + sqrt(1-alpha) * noise
/// - The path from noise to image is curved (follows a complex schedule)
/// - Needs many steps to follow the curved path accurately
///
/// Flow matching (rectified flow):
/// - Uses simple linear interpolation: x_t = (1-t) * image + t * noise
/// - The path from noise to image is a straight line
/// - Can traverse the straight path in fewer steps (often 20-50 vs 50-100+)
/// - The model predicts velocity v = noise - image (direction to move)
///
/// ODE sampling step:
/// - Start at x_1 (pure noise)
/// - At each step: x_{t-dt} = x_t - dt * v(x_t, t)
/// - After all steps: arrive at x_0 (clean image)
///
/// Used by:
/// - Stable Diffusion 3 (SD3)
/// - FLUX.1
/// - Stable Diffusion 3.5
/// </para>
/// <para>
/// <b>Reference:</b> Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023;
/// Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023
/// </para>
/// </remarks>
public sealed class FlowMatchingScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// Sigma values for each timestep (optional noise scaling for stochastic sampling).
    /// </summary>
    private Vector<T>? _sigmas;

    /// <summary>
    /// The continuous time values corresponding to the discrete timesteps.
    /// </summary>
    /// <remarks>
    /// Maps integer timesteps to continuous t in [0, 1] where t=0 is clean and t=1 is noise.
    /// </remarks>
    private Vector<T> _timeValues;

    /// <summary>
    /// Initializes a new flow matching scheduler with rectified flow defaults.
    /// </summary>
    /// <param name="config">
    /// Scheduler configuration. Use <see cref="SchedulerConfig{T}.CreateRectifiedFlow"/> for SD3/FLUX defaults.
    /// The prediction type should be VPrediction for rectified flow models.
    /// </param>
    /// <example>
    /// <code>
    /// // Create for Stable Diffusion 3
    /// var scheduler = new FlowMatchingScheduler&lt;float&gt;(
    ///     SchedulerConfig&lt;float&gt;.CreateRectifiedFlow());
    ///
    /// // Set up 28 inference steps (recommended for SD3)
    /// scheduler.SetTimesteps(28);
    ///
    /// // Denoising loop
    /// var eta = 0f; // Deterministic
    /// foreach (var t in scheduler.Timesteps)
    /// {
    ///     var velocity = model.Predict(noisySample, t);
    ///     noisySample = scheduler.Step(velocity, t, noisySample, eta);
    /// }
    /// </code>
    /// </example>
    public FlowMatchingScheduler(SchedulerConfig<T> config)
        : base(config)
    {
        // Pre-compute continuous time values for all training timesteps
        _timeValues = ComputeTimeValues(config.TrainTimesteps);
    }

    /// <summary>
    /// Creates a flow matching scheduler with default SD3/FLUX configuration.
    /// </summary>
    /// <returns>A new flow matching scheduler ready for use.</returns>
    public static FlowMatchingScheduler<T> CreateDefault()
    {
        return new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow());
    }

    /// <summary>
    /// Sets up linearly-spaced inference timesteps for flow matching.
    /// </summary>
    /// <param name="inferenceSteps">Number of denoising steps. 20-50 recommended for quality, 4-8 for speed.</param>
    /// <remarks>
    /// <para>
    /// Flow matching uses linearly spaced timesteps from T-1 down to 0, with optional
    /// time shifting for improved sampling quality.
    /// </para>
    /// </remarks>
    public override void SetTimesteps(int inferenceSteps)
    {
        if (inferenceSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(inferenceSteps), "Inference steps must be positive.");
        if (inferenceSteps > Config.TrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(inferenceSteps),
                $"Inference steps ({inferenceSteps}) cannot exceed training timesteps ({Config.TrainTimesteps}).");

        // Linearly spaced timesteps from T-1 to 0
        var timestepList = new List<int>();
        double stepSize = (double)(Config.TrainTimesteps - 1) / inferenceSteps;

        for (int i = 0; i < inferenceSteps; i++)
        {
            int timestep = (int)Math.Round((Config.TrainTimesteps - 1) - i * stepSize);
            timestep = Math.Max(0, Math.Min(timestep, Config.TrainTimesteps - 1));
            timestepList.Add(timestep);
        }

        // Use reflection to set the private _timesteps field via the base class
        // Instead, we call the base which sets it, then we'll compute sigmas
        base.SetTimesteps(inferenceSteps);

        // Compute sigmas for optional stochastic sampling
        _sigmas = ComputeSigmas(inferenceSteps);
    }

    /// <summary>
    /// Performs one flow matching ODE step (Euler method).
    /// </summary>
    /// <param name="modelOutput">The model's velocity prediction v(x_t, t).</param>
    /// <param name="timestep">The current timestep.</param>
    /// <param name="sample">The current sample x_t.</param>
    /// <param name="eta">Stochasticity parameter (0 = deterministic ODE, >0 = stochastic SDE).</param>
    /// <param name="noise">Optional noise for stochastic sampling.</param>
    /// <returns>The denoised sample at the previous timestep.</returns>
    /// <remarks>
    /// <para>
    /// The Euler ODE step for rectified flow:
    ///   x_{t-dt} = x_t - dt * v(x_t, t)
    ///
    /// where dt is the step size in continuous time and v is the predicted velocity.
    ///
    /// For v-prediction (rectified flow), the velocity v = noise - x_0, so:
    ///   x_0 = x_t - t * v  (predicted clean sample)
    ///   x_{t-dt} = (1 - (t-dt)) * x_0 + (t-dt) * noise
    ///
    /// This simplifies to the Euler step: x_{t-dt} = x_t - dt * v
    /// </para>
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        // Get continuous time t for current and previous timestep
        T currentT = GetContinuousTime(timestep);
        T prevT = GetPreviousContinuousTime(timestep);

        // dt = t_prev - t_current (negative because we go from t=1 to t=0)
        T dt = NumOps.Subtract(prevT, currentT);

        Vector<T> prevSample;

        if (Config.PredictionType == DiffusionPredictionType.VPrediction)
        {
            // Rectified flow: model predicts velocity v = noise - x_0
            // Euler step: x_{t-dt} = x_t + dt * v  (dt is negative, so this denoises)
            // But since dt = prevT - currentT is negative when going from high to low t,
            // and v points from data to noise, we want: x_prev = x_t + dt * v
            var scaledVelocity = Engine.Multiply(modelOutput, dt);
            prevSample = Engine.Add(sample, scaledVelocity);
        }
        else if (Config.PredictionType == DiffusionPredictionType.Epsilon)
        {
            // If predicting noise, convert to velocity: v = (x_t - x_0) / t
            // x_0 = x_t - t * noise (approximately, using flow matching formulation)
            // Then apply Euler step
            var tScaled = Engine.Multiply(modelOutput, currentT);
            var predictedOriginal = Engine.Subtract(sample, tScaled);
            predictedOriginal = ClipSampleIfNeeded(predictedOriginal);

            // Compute velocity from predicted original: v = (noise - x_0) ≈ (x_t - x_0) / t
            // Euler step: x_prev = (1 - t_prev) * x_0 + t_prev * noise
            var oneMinusTprev = NumOps.Subtract(NumOps.One, prevT);
            var signalPart = Engine.Multiply(predictedOriginal, oneMinusTprev);
            var noisePart = Engine.Multiply(modelOutput, prevT);
            prevSample = Engine.Add(signalPart, noisePart);
        }
        else // DiffusionPredictionType.Sample
        {
            // Model directly predicts x_0
            var predictedOriginal = ClipSampleIfNeeded(modelOutput);

            // Compute noise from x_t and x_0: noise = (x_t - (1-t)*x_0) / t
            // Then: x_prev = (1 - t_prev) * x_0 + t_prev * noise
            if (NumOps.GreaterThan(currentT, NumOps.FromDouble(1e-6)))
            {
                var oneMinusT = NumOps.Subtract(NumOps.One, currentT);
                var signalComponent = Engine.Multiply(predictedOriginal, oneMinusT);
                var noiseComponent = Engine.Subtract(sample, signalComponent);
                var estimatedNoise = Engine.Divide(noiseComponent, currentT);

                var oneMinusTprev = NumOps.Subtract(NumOps.One, prevT);
                var prevSignal = Engine.Multiply(predictedOriginal, oneMinusTprev);
                var prevNoise = Engine.Multiply(estimatedNoise, prevT);
                prevSample = Engine.Add(prevSignal, prevNoise);
            }
            else
            {
                // At t≈0, we're already at the clean sample
                prevSample = predictedOriginal;
            }
        }

        // Optional stochastic noise injection
        var etaDouble = NumOps.ToDouble(eta);
        if (etaDouble > 0 && noise != null)
        {
            var noiseScale = NumOps.Multiply(eta, NumOps.FromDouble(Math.Abs(NumOps.ToDouble(NumOps.Subtract(prevT, currentT)))));
            var stochasticNoise = Engine.Multiply(noise, noiseScale);
            prevSample = Engine.Add(prevSample, stochasticNoise);
        }

        return prevSample;
    }

    /// <summary>
    /// Adds noise using flow matching linear interpolation.
    /// </summary>
    /// <param name="originalSample">The clean sample x_0.</param>
    /// <param name="noise">The target noise.</param>
    /// <param name="timestep">The timestep determining noise level.</param>
    /// <returns>The noisy sample x_t = (1-t)*x_0 + t*noise.</returns>
    /// <remarks>
    /// <para>
    /// Flow matching uses simple linear interpolation between the clean sample and noise,
    /// unlike DDPM which uses sqrt-based scaling. This creates a straight interpolation path.
    /// </para>
    /// </remarks>
    public override Vector<T> AddNoise(Vector<T> originalSample, Vector<T> noise, int timestep)
    {
        if (originalSample == null)
            throw new ArgumentNullException(nameof(originalSample));
        if (noise == null)
            throw new ArgumentNullException(nameof(noise));
        if (originalSample.Length != noise.Length)
            throw new ArgumentException("Original sample and noise must have the same length.", nameof(noise));
        if (timestep < 0 || timestep >= Config.TrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(timestep));

        // Flow matching: x_t = (1 - t) * x_0 + t * noise
        T t = GetContinuousTime(timestep);
        T oneMinusT = NumOps.Subtract(NumOps.One, t);

        var signalPart = Engine.Multiply(originalSample, oneMinusT);
        var noisePart = Engine.Multiply(noise, t);
        return Engine.Add(signalPart, noisePart);
    }

    /// <inheritdoc />
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["scheduler_type"] = "FlowMatching";
        return state;
    }

    /// <summary>
    /// Converts an integer timestep to continuous time t in [0, 1].
    /// </summary>
    /// <param name="timestep">Integer timestep in [0, TrainTimesteps-1].</param>
    /// <returns>Continuous time t where 0 = clean sample and 1 = pure noise.</returns>
    private T GetContinuousTime(int timestep)
    {
        if (timestep < 0) return NumOps.Zero;
        if (timestep >= _timeValues.Length) return NumOps.One;
        return _timeValues[timestep];
    }

    /// <summary>
    /// Gets the continuous time for the previous timestep in the inference schedule.
    /// </summary>
    /// <param name="currentTimestep">The current integer timestep.</param>
    /// <returns>The continuous time for the next step closer to t=0.</returns>
    private T GetPreviousContinuousTime(int currentTimestep)
    {
        var timesteps = Timesteps;

        // Find the current timestep in the schedule
        int idx = -1;
        for (int i = 0; i < timesteps.Length; i++)
        {
            if (timesteps[i] == currentTimestep)
            {
                idx = i;
                break;
            }
        }

        // If this is the last timestep or not found, return t=0
        if (idx < 0 || idx >= timesteps.Length - 1)
            return NumOps.Zero;

        // Return the time for the next timestep in the schedule
        return GetContinuousTime(timesteps[idx + 1]);
    }

    /// <summary>
    /// Pre-computes continuous time values for all training timesteps.
    /// </summary>
    /// <param name="trainTimesteps">Total number of training timesteps.</param>
    /// <returns>Vector of continuous time values t = timestep / (trainTimesteps - 1).</returns>
    private static Vector<T> ComputeTimeValues(int trainTimesteps)
    {
        var timeValues = new Vector<T>(trainTimesteps);
        double scale = 1.0 / (trainTimesteps - 1);

        for (int i = 0; i < trainTimesteps; i++)
        {
            timeValues[i] = NumOps.FromDouble(i * scale);
        }

        return timeValues;
    }

    /// <summary>
    /// Computes sigma values for optional stochastic sampling.
    /// </summary>
    /// <param name="inferenceSteps">Number of inference steps.</param>
    /// <returns>Sigma values derived from the continuous time schedule.</returns>
    private Vector<T> ComputeSigmas(int inferenceSteps)
    {
        var sigmas = new Vector<T>(inferenceSteps + 1);
        var timesteps = Timesteps;

        for (int i = 0; i < timesteps.Length && i < inferenceSteps; i++)
        {
            // Sigma is proportional to t (noise level) in flow matching
            sigmas[i] = GetContinuousTime(timesteps[i]);
        }

        // Final sigma = 0 (clean sample)
        sigmas[inferenceSteps] = NumOps.Zero;

        return sigmas;
    }
}
