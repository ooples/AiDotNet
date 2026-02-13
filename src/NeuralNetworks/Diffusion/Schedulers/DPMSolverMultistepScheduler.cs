using AiDotNet.Enums;

namespace AiDotNet.NeuralNetworks.Diffusion.Schedulers;

/// <summary>
/// DPM-Solver++ multistep scheduler for fast diffusion model sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DPM-Solver++ is a high-order ODE solver specifically designed for diffusion models.
/// It achieves state-of-the-art sampling quality with very few steps (10-25) by using
/// multi-step methods that leverage history of previous model evaluations.
/// </para>
/// <para>
/// <b>For Beginners:</b> DPM-Solver++ is one of the fastest high-quality samplers.
///
/// Think of it like navigating with a GPS that remembers your path:
/// - Euler: Looks at current position only, takes simple steps
/// - DPM-Solver++: Remembers previous positions, predicts better next steps
///
/// Key characteristics:
/// - Very fast: Good quality in just 15-25 steps
/// - Multi-order: Uses 1st, 2nd, or 3rd order methods adaptively
/// - Deterministic: Same seed always gives same result
/// - Widely adopted: Used as default in many Stable Diffusion implementations
///
/// The "multistep" means it stores previous model outputs and uses them
/// to make more accurate predictions, similar to Adams-Bashforth methods
/// in numerical analysis.
/// </para>
/// <para>
/// <b>Reference:</b> Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", 2022
/// </para>
/// </remarks>
public sealed class DPMSolverMultistepScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// History of model outputs for multi-step methods.
    /// </summary>
    private readonly List<Vector<T>> _modelOutputHistory = new();

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
    /// Maximum order of the solver (1, 2, or 3).
    /// </summary>
    private readonly int _solverOrder;

    /// <summary>
    /// Initializes a new instance of the DPM-Solver++ multistep scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    /// <param name="solverOrder">Maximum order of the solver (1-3). Higher = more accurate per step. Default: 2.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when solverOrder is not 1, 2, or 3.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a DPM-Solver++ scheduler for fast, high-quality sampling.
    /// Order 2 is recommended for the best speed/quality trade-off.
    /// </para>
    /// <example>
    /// <code>
    /// var config = SchedulerConfig&lt;double&gt;.CreateStableDiffusion();
    /// var scheduler = new DPMSolverMultistepScheduler&lt;double&gt;(config, solverOrder: 2);
    /// scheduler.SetTimesteps(20);
    /// </code>
    /// </example>
    /// </remarks>
    public DPMSolverMultistepScheduler(SchedulerConfig<T> config, int solverOrder = 2) : base(config)
    {
        if (solverOrder < 1 || solverOrder > 3)
            throw new ArgumentOutOfRangeException(nameof(solverOrder), "Solver order must be 1, 2, or 3.");

        _solverOrder = solverOrder;
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

            // lambda = log(alpha_t / sigma_t) = log(SNR) / 2
            T ratio = NumOps.Divide(sqrtAlpha, sigma);
            // Use log approximation: log(x) ≈ (x-1) - (x-1)^2/2 + ... for stability
            T ratioDouble = ratio;
            _lambdas[i] = NumOps.FromDouble(Math.Log(Math.Max(NumOps.ToDouble(ratioDouble), 1e-10)));
        }
    }

    /// <summary>
    /// Resets the scheduler state for a new generation run.
    /// </summary>
    private void ResetState()
    {
        _modelOutputHistory.Clear();
        _stepCounter = 0;
    }

    /// <summary>
    /// Performs one DPM-Solver++ multistep denoising step.
    /// </summary>
    /// <param name="modelOutput">The model's noise prediction.</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">Not used in DPM-Solver++ (deterministic). Included for interface compatibility.</param>
    /// <param name="noise">Not used in DPM-Solver++ (deterministic). Included for interface compatibility.</param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <remarks>
    /// <para>
    /// DPM-Solver++ multistep:
    /// 1. Convert model output to data prediction (x_0 prediction form)
    /// 2. Store in history buffer
    /// 3. Apply 1st, 2nd, or 3rd order update based on available history
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

        // Store in history (keep last _solverOrder entries)
        _modelOutputHistory.Add(CopyVector(dataPred));
        while (_modelOutputHistory.Count > _solverOrder + 1)
        {
            _modelOutputHistory.RemoveAt(0);
        }

        // Determine effective order (can't use higher order than available history)
        int effectiveOrder = Math.Min(_stepCounter + 1, _solverOrder);
        effectiveOrder = Math.Min(effectiveOrder, _modelOutputHistory.Count);

        // Ensure we have a valid next step index
        int nextStepIndex = Math.Min(stepIndex + 1, Timesteps.Length - 1);
        if (stepIndex >= Timesteps.Length - 1)
        {
            // Last step - return the data prediction directly
            _stepCounter++;
            return dataPred;
        }

        // Apply the appropriate order update
        Vector<T> prevSample;
        switch (effectiveOrder)
        {
            case 1:
                prevSample = FirstOrderUpdate(sample, stepIndex, nextStepIndex);
                break;
            case 2:
                prevSample = SecondOrderUpdate(sample, stepIndex, nextStepIndex);
                break;
            default:
                prevSample = ThirdOrderUpdate(sample, stepIndex, nextStepIndex);
                break;
        }

        _stepCounter++;
        return prevSample;
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
                return modelOutput;

            case DiffusionPredictionType.VPrediction:
                var vX0 = Engine.Multiply(sample, sqrtAlphaCumprod);
                var vEps = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprod);
                return Engine.Subtract(vX0, vEps);

            default: // Epsilon
                var noiseTerm = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprod);
                var numerator = Engine.Subtract(sample, noiseTerm);
                return Engine.Divide(numerator, sqrtAlphaCumprod);
        }
    }

    /// <summary>
    /// First-order DPM-Solver++ update (equivalent to DDIM).
    /// </summary>
    private Vector<T> FirstOrderUpdate(Vector<T> sample, int stepIndex, int nextStepIndex)
    {
        T lambda_s = _lambdas![stepIndex];
        T lambda_t = _lambdas[nextStepIndex];
        T alpha_t = _alphaTs![nextStepIndex];
        T sigma_t = _sigmaTs![nextStepIndex];
        T sigma_s = _sigmaTs[stepIndex];

        // h = lambda_t - lambda_s
        T h = NumOps.Subtract(lambda_t, lambda_s);

        // DPM-Solver++ first-order: x_t = (sigma_t/sigma_s) * x_s + alpha_t * (exp(-h) - 1) * D_0
        T sigmaRatio = NumOps.Divide(sigma_t, sigma_s);
        T expNegH = NumOps.FromDouble(Math.Exp(-NumOps.ToDouble(h)));
        T coeff = NumOps.Multiply(alpha_t, NumOps.Subtract(expNegH, NumOps.One));

        var scaledSample = Engine.Multiply(sample, sigmaRatio);
        var dataTerm = Engine.Multiply(_modelOutputHistory[^1], coeff);
        return Engine.Add(scaledSample, dataTerm);
    }

    /// <summary>
    /// Second-order DPM-Solver++ update using one previous model output.
    /// </summary>
    private Vector<T> SecondOrderUpdate(Vector<T> sample, int stepIndex, int nextStepIndex)
    {
        if (_modelOutputHistory.Count < 2)
            return FirstOrderUpdate(sample, stepIndex, nextStepIndex);

        T lambda_s = _lambdas![stepIndex];
        T lambda_t = _lambdas[nextStepIndex];
        T alpha_t = _alphaTs![nextStepIndex];
        T sigma_t = _sigmaTs![nextStepIndex];
        T sigma_s = _sigmaTs[stepIndex];

        T h = NumOps.Subtract(lambda_t, lambda_s);

        T sigmaRatio = NumOps.Divide(sigma_t, sigma_s);
        T expNegH = NumOps.FromDouble(Math.Exp(-NumOps.ToDouble(h)));
        T coeff1 = NumOps.Multiply(alpha_t, NumOps.Subtract(expNegH, NumOps.One));

        // Second-order correction: (exp(-h) - 1 + h) / (2h)
        T correction = NumOps.FromDouble((Math.Exp(-NumOps.ToDouble(h)) - 1.0 + NumOps.ToDouble(h)) / (2.0 * NumOps.ToDouble(h)));
        T coeff2 = NumOps.Multiply(alpha_t, correction);

        // D_0 = latest data prediction, D_1 = difference from previous
        var d0 = _modelOutputHistory[^1];
        var d1 = Engine.Subtract(_modelOutputHistory[^1], _modelOutputHistory[^2]);

        var scaledSample = Engine.Multiply(sample, sigmaRatio);
        var term1 = Engine.Multiply(d0, coeff1);
        var term2 = Engine.Multiply(d1, coeff2);
        var result = Engine.Add(scaledSample, term1);
        return Engine.Add(result, term2);
    }

    /// <summary>
    /// Third-order DPM-Solver++ update using two previous model outputs.
    /// </summary>
    private Vector<T> ThirdOrderUpdate(Vector<T> sample, int stepIndex, int nextStepIndex)
    {
        if (_modelOutputHistory.Count < 3)
            return SecondOrderUpdate(sample, stepIndex, nextStepIndex);

        T lambda_s = _lambdas![stepIndex];
        T lambda_t = _lambdas[nextStepIndex];
        T alpha_t = _alphaTs![nextStepIndex];
        T sigma_t = _sigmaTs![nextStepIndex];
        T sigma_s = _sigmaTs[stepIndex];

        T h = NumOps.Subtract(lambda_t, lambda_s);
        double hVal = NumOps.ToDouble(h);

        T sigmaRatio = NumOps.Divide(sigma_t, sigma_s);
        T expNegH = NumOps.FromDouble(Math.Exp(-hVal));

        // Coefficients for 3rd order
        T coeff1 = NumOps.Multiply(alpha_t, NumOps.Subtract(expNegH, NumOps.One));
        T coeff2 = NumOps.FromDouble(NumOps.ToDouble(alpha_t) * (Math.Exp(-hVal) - 1.0 + hVal) / (2.0 * hVal));
        T coeff3 = NumOps.FromDouble(NumOps.ToDouble(alpha_t) * (Math.Exp(-hVal) - 1.0 + hVal - 0.5 * hVal * hVal) / (6.0 * hVal * hVal));

        var d0 = _modelOutputHistory[^1];
        var d1 = Engine.Subtract(_modelOutputHistory[^1], _modelOutputHistory[^2]);
        var d2Prev = Engine.Subtract(_modelOutputHistory[^2], _modelOutputHistory[^3]);
        var d2 = Engine.Subtract(d1, d2Prev);

        var scaledSample = Engine.Multiply(sample, sigmaRatio);
        var term1 = Engine.Multiply(d0, coeff1);
        var term2 = Engine.Multiply(d1, coeff2);
        var term3 = Engine.Multiply(d2, coeff3);
        var result = Engine.Add(scaledSample, term1);
        result = Engine.Add(result, term2);
        return Engine.Add(result, term3);
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
        state["solver_order"] = _solverOrder;
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
