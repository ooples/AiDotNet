using AiDotNet.Enums;

namespace AiDotNet.Diffusion;

/// <summary>
/// UniPC (Unified Predictor-Corrector) scheduler for fast, high-quality diffusion sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// UniPC combines predictor and corrector steps into a unified framework, achieving
/// superior sampling quality with fewer function evaluations compared to pure predictor
/// methods like DDIM or DPM-Solver++.
/// </para>
/// <para>
/// <b>For Beginners:</b> UniPC is like a "guess and check" approach to denoising.
///
/// Most schedulers just predict the next step (predict only):
/// - Step 1: Predict the cleaner image → use as-is
///
/// UniPC adds a correction step for better accuracy:
/// - Step 1: Predict the cleaner image (predictor step)
/// - Step 2: Check how good the prediction was and improve it (corrector step)
///
/// This two-phase approach means:
/// - Better quality at the same number of steps (e.g., 10-step UniPC ≈ 15-step DDIM)
/// - Or same quality with fewer steps (faster generation)
///
/// Key characteristics:
/// - Combines predictor-corrector methodology with multi-step methods
/// - Supports orders 1-3 (higher = more accurate per step)
/// - Deterministic by default
/// - Particularly effective at very low step counts (5-15 steps)
///
/// Used by:
/// - ComfyUI, A1111, and many Stable Diffusion UIs as an alternative scheduler
/// - Effective with SD 1.5, SDXL, and other diffusion models
/// </para>
/// <para>
/// <b>Reference:</b> Zhao et al., "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models", NeurIPS 2023
/// </para>
/// </remarks>
public sealed class UniPCScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// History of data predictions (x_0 form) for multi-step methods.
    /// </summary>
    private readonly List<Vector<T>> _dataPredictionHistory = new();

    /// <summary>
    /// Lambda values (log-SNR) for each inference timestep.
    /// </summary>
    private Vector<T>? _lambdas;

    /// <summary>
    /// Alpha_t values for inference timesteps.
    /// </summary>
    private Vector<T>? _alphaTs;

    /// <summary>
    /// Sigma_t values for inference timesteps.
    /// </summary>
    private Vector<T>? _sigmaTs;

    /// <summary>
    /// Current step counter for tracking multi-step order.
    /// </summary>
    private int _stepCounter;

    /// <summary>
    /// Maximum order of the predictor-corrector solver (1-3).
    /// </summary>
    private readonly int _solverOrder;

    /// <summary>
    /// Whether to apply the corrector step after the predictor.
    /// </summary>
    private readonly bool _useCorrectorStep;

    /// <summary>
    /// Initializes a new instance of the UniPC scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    /// <param name="solverOrder">Maximum order of the solver (1-3). Higher = more accurate per step. Default: 2.</param>
    /// <param name="useCorrectorStep">Whether to enable the corrector step. Default: true.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when solverOrder is not 1, 2, or 3.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a UniPC scheduler for fast, high-quality sampling.
    /// Order 2 with corrector enabled is recommended for the best results.
    /// </para>
    /// <example>
    /// <code>
    /// var config = SchedulerConfig&lt;double&gt;.CreateStableDiffusion();
    /// var scheduler = new UniPCScheduler&lt;double&gt;(config, solverOrder: 2, useCorrectorStep: true);
    /// scheduler.SetTimesteps(15);
    /// </code>
    /// </example>
    /// </remarks>
    public UniPCScheduler(SchedulerConfig<T> config, int solverOrder = 2, bool useCorrectorStep = true)
        : base(config)
    {
        if (solverOrder < 1 || solverOrder > 3)
            throw new ArgumentOutOfRangeException(nameof(solverOrder), "Solver order must be 1, 2, or 3.");

        _solverOrder = solverOrder;
        _useCorrectorStep = useCorrectorStep;
        _stepCounter = 0;
    }

    /// <summary>
    /// Sets up the inference timesteps and computes lambda/sigma/alpha schedules.
    /// </summary>
    /// <param name="inferenceSteps">Number of denoising steps to use during inference.</param>
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        ComputeScheduleValues();
        ResetState();
    }

    /// <summary>
    /// Performs one UniPC predictor-corrector denoising step.
    /// </summary>
    /// <param name="modelOutput">The model's noise prediction.</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">Stochasticity parameter (0 = deterministic). Included for interface compatibility.</param>
    /// <param name="noise">Optional noise for stochastic sampling.</param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <remarks>
    /// <para>
    /// UniPC step:
    /// 1. Convert model output to data prediction (x_0 form)
    /// 2. Store in history buffer for multi-step method
    /// 3. Apply predictor step (multi-step extrapolation)
    /// 4. Optionally apply corrector step (refine using prediction)
    /// </para>
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        if (_lambdas == null || _alphaTs == null || _sigmaTs == null)
            throw new InvalidOperationException("Schedule not initialized. Call SetTimesteps() before Step().");

        int stepIndex = FindTimestepIndex(timestep);

        // Convert model output to data prediction (x_0 form)
        var dataPred = ConvertModelOutputToDataPrediction(modelOutput, timestep, sample);

        // Store in history (keep last _solverOrder + 1 entries)
        _dataPredictionHistory.Add(CopyVector(dataPred));
        while (_dataPredictionHistory.Count > _solverOrder + 1)
        {
            _dataPredictionHistory.RemoveAt(0);
        }

        // Last step: return the data prediction directly
        if (stepIndex >= Timesteps.Length - 1)
        {
            _stepCounter++;
            return dataPred;
        }

        int nextStepIndex = stepIndex + 1;

        // Determine effective order based on available history
        int effectiveOrder = Math.Min(_stepCounter + 1, _solverOrder);
        effectiveOrder = Math.Min(effectiveOrder, _dataPredictionHistory.Count);

        // Predictor step
        var predictedSample = PredictorStep(sample, stepIndex, nextStepIndex, effectiveOrder);

        // Corrector step (optional refinement)
        if (_useCorrectorStep && effectiveOrder >= 2 && _dataPredictionHistory.Count >= 2)
        {
            predictedSample = CorrectorStep(predictedSample, sample, stepIndex, nextStepIndex, effectiveOrder);
        }

        // Optional stochastic noise injection
        var etaDouble = NumOps.ToDouble(eta);
        if (etaDouble > 0 && noise != null)
        {
            T lambda_s = _lambdas[stepIndex];
            T lambda_t = _lambdas[nextStepIndex];
            T h = NumOps.Subtract(lambda_t, lambda_s);
            var noiseScale = NumOps.Multiply(eta, NumOps.FromDouble(Math.Abs(NumOps.ToDouble(h))));
            var stochasticNoise = Engine.Multiply(noise, noiseScale);
            predictedSample = Engine.Add(predictedSample, stochasticNoise);
        }

        _stepCounter++;
        return predictedSample;
    }

    /// <summary>
    /// Predictor step: multi-step extrapolation to estimate the next sample.
    /// </summary>
    /// <param name="sample">The current sample.</param>
    /// <param name="stepIndex">Current step index in the schedule.</param>
    /// <param name="nextStepIndex">Next step index in the schedule.</param>
    /// <param name="order">The effective order to use.</param>
    /// <returns>The predicted sample at the next timestep.</returns>
    private Vector<T> PredictorStep(Vector<T> sample, int stepIndex, int nextStepIndex, int order)
    {
        T lambda_s = _lambdas![stepIndex];
        T lambda_t = _lambdas[nextStepIndex];
        T alpha_t = _alphaTs![nextStepIndex];
        T sigma_t = _sigmaTs![nextStepIndex];
        T sigma_s = _sigmaTs[stepIndex];

        T h = NumOps.Subtract(lambda_t, lambda_s);
        double hVal = NumOps.ToDouble(h);

        T sigmaRatio = NumOps.Divide(sigma_t, sigma_s);

        // First-order term: always present
        T expNegH = NumOps.FromDouble(Math.Exp(-hVal));
        T coeff0 = NumOps.Multiply(alpha_t, NumOps.Subtract(expNegH, NumOps.One));

        var d0 = _dataPredictionHistory[^1];
        var result = Engine.Add(Engine.Multiply(sample, sigmaRatio), Engine.Multiply(d0, coeff0));

        if (order >= 2 && _dataPredictionHistory.Count >= 2)
        {
            // Second-order correction using Taylor expansion
            T coeff1 = NumOps.FromDouble(
                NumOps.ToDouble(alpha_t) * (Math.Exp(-hVal) - 1.0 + hVal) / (2.0 * hVal));

            var d1 = Engine.Subtract(_dataPredictionHistory[^1], _dataPredictionHistory[^2]);
            result = Engine.Add(result, Engine.Multiply(d1, coeff1));
        }

        if (order >= 3 && _dataPredictionHistory.Count >= 3)
        {
            // Third-order correction
            T coeff2 = NumOps.FromDouble(
                NumOps.ToDouble(alpha_t) * (Math.Exp(-hVal) - 1.0 + hVal - 0.5 * hVal * hVal) / (6.0 * hVal * hVal));

            var d1 = Engine.Subtract(_dataPredictionHistory[^1], _dataPredictionHistory[^2]);
            var d1Prev = Engine.Subtract(_dataPredictionHistory[^2], _dataPredictionHistory[^3]);
            var d2 = Engine.Subtract(d1, d1Prev);
            result = Engine.Add(result, Engine.Multiply(d2, coeff2));
        }

        return result;
    }

    /// <summary>
    /// Corrector step: refines the predictor output using interpolation error estimation.
    /// </summary>
    /// <param name="predictedSample">The sample from the predictor step.</param>
    /// <param name="currentSample">The original sample before prediction.</param>
    /// <param name="stepIndex">Current step index in the schedule.</param>
    /// <param name="nextStepIndex">Next step index in the schedule.</param>
    /// <param name="order">The effective order to use.</param>
    /// <returns>The corrected sample.</returns>
    /// <remarks>
    /// <para>
    /// The corrector step in UniPC estimates the local truncation error of the predictor
    /// and uses it to refine the prediction. This is based on the difference between
    /// the multi-step predictor output and a single-step estimate at the predicted point.
    /// </para>
    /// </remarks>
    private Vector<T> CorrectorStep(Vector<T> predictedSample, Vector<T> currentSample,
        int stepIndex, int nextStepIndex, int order)
    {
        T lambda_s = _lambdas![stepIndex];
        T lambda_t = _lambdas![nextStepIndex];
        T alpha_t = _alphaTs![nextStepIndex];
        T alpha_s = _alphaTs[stepIndex];
        T sigma_t = _sigmaTs![nextStepIndex];
        T sigma_s = _sigmaTs[stepIndex];

        T h = NumOps.Subtract(lambda_t, lambda_s);
        double hVal = NumOps.ToDouble(h);

        // Estimate x_0 at the predicted point using the latest and previous data predictions
        // This approximates what a model evaluation at the predicted point would give
        var latestPred = _dataPredictionHistory[^1];
        var prevPred = _dataPredictionHistory[^2];

        // Compute the correction term based on the difference in consecutive predictions
        // The corrector refines based on the estimated curvature of the data prediction trajectory
        var predDiff = Engine.Subtract(latestPred, prevPred);

        // Correction coefficient: weight based on step size
        // c_corr = alpha_t * h * (exp(-h) - 1 + h) / (2 * h^2)
        // This is the leading error term of the first-order predictor
        double correctionWeight;
        if (Math.Abs(hVal) < 1e-8)
        {
            correctionWeight = 0; // Avoid division by zero
        }
        else
        {
            correctionWeight = NumOps.ToDouble(alpha_t) *
                (Math.Exp(-hVal) - 1.0 + hVal) / (2.0 * hVal);
        }

        // Scale correction by the inverse of the order to avoid over-correction
        correctionWeight /= order;

        var correctionCoeff = NumOps.FromDouble(correctionWeight);
        var correction = Engine.Multiply(predDiff, correctionCoeff);

        // Apply correction to the predicted sample
        return Engine.Add(predictedSample, correction);
    }

    /// <summary>
    /// Converts model output to data prediction (x_0 form) based on prediction type.
    /// </summary>
    private Vector<T> ConvertModelOutputToDataPrediction(Vector<T> modelOutput, int timestep, Vector<T> sample)
    {
        T alphaCumprod = AlphasCumulativeProduct[timestep];
        T sqrtAlphaCumprod = NumOps.Sqrt(alphaCumprod);
        T sqrtOneMinusAlphaCumprod = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));

        switch (Config.PredictionType)
        {
            case DiffusionPredictionType.Sample:
                return ClipSampleIfNeeded(modelOutput);

            case DiffusionPredictionType.VPrediction:
                var vX0 = Engine.Multiply(sample, sqrtAlphaCumprod);
                var vEps = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprod);
                return ClipSampleIfNeeded(Engine.Subtract(vX0, vEps));

            default: // Epsilon
                var noiseTerm = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprod);
                var numerator = Engine.Subtract(sample, noiseTerm);
                return ClipSampleIfNeeded(Engine.Divide(numerator, sqrtAlphaCumprod));
        }
    }

    /// <summary>
    /// Computes lambda (log-SNR), alpha_t, and sigma_t values for the inference schedule.
    /// </summary>
    private void ComputeScheduleValues()
    {
        var timesteps = Timesteps;
        int len = timesteps.Length;

        _lambdas = new Vector<T>(len);
        _alphaTs = new Vector<T>(len);
        _sigmaTs = new Vector<T>(len);

        for (int i = 0; i < len; i++)
        {
            T alphaCumprod = AlphasCumulativeProduct[timesteps[i]];
            T sqrtAlpha = NumOps.Sqrt(alphaCumprod);
            T oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaCumprod);
            T sigma = NumOps.Sqrt(NumOps.Divide(oneMinusAlpha, alphaCumprod));

            _alphaTs[i] = sqrtAlpha;
            _sigmaTs[i] = sigma;

            T ratio = NumOps.Divide(sqrtAlpha, sigma);
            _lambdas[i] = NumOps.FromDouble(Math.Log(Math.Max(NumOps.ToDouble(ratio), 1e-10)));
        }
    }

    /// <summary>
    /// Resets the scheduler state for a new generation run.
    /// </summary>
    private void ResetState()
    {
        _dataPredictionHistory.Clear();
        _stepCounter = 0;
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

        // Find closest
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
    /// Creates a copy of a vector.
    /// </summary>
    private static Vector<T> CopyVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            copy[i] = source[i];
        }
        return copy;
    }

    /// <inheritdoc />
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["scheduler_type"] = "UniPC";
        state["solver_order"] = _solverOrder;
        state["use_corrector"] = _useCorrectorStep;
        state["step_counter"] = _stepCounter;
        return state;
    }

    /// <inheritdoc />
    public override void LoadState(Dictionary<string, object> state)
    {
        base.LoadState(state);
        ResetState();
    }
}
