using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Schedulers;

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
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public sealed class UniPCScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// History of data predictions (x_0 form) for multi-step methods.
    /// </summary>
    private readonly List<Vector<T>> _dataPredictionHistory = new();

    /// <summary>
    /// Lambda values (half log-SNR) of the entries in <see cref="_dataPredictionHistory"/>.
    /// </summary>
    private readonly List<T> _lambdaHistory = new();

    /// <summary>
    /// The sample the last predictor started from (the corrector's x_s0).
    /// </summary>
    private Vector<T>? _lastSample;

    /// <summary>
    /// The order used by the last predictor step (reused by the following corrector).
    /// </summary>
    private int _thisOrder = 1;

    /// <summary>
    /// Number of completed warm-up steps (capped at the solver order).
    /// </summary>
    private int _lowerOrderNums;

    /// <summary>
    /// Lambda values (log-SNR) for each inference timestep.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private Vector<T> _lambdas = new Vector<T>(0);

    /// <summary>
    /// Alpha_t values for inference timesteps.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private Vector<T> _alphaTs = new Vector<T>(0);

    /// <summary>
    /// Sigma_t values for inference timesteps.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private Vector<T> _sigmaTs = new Vector<T>(0);

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
    /// Performs one UniPC step: the UniC corrector refines the previous predictor output with the new model
    /// evaluation, then the UniP predictor advances to the next timestep.
    /// </summary>
    /// <remarks>
    /// Follows Zhao et al. (2023) and diffusers' UniPCMultistepScheduler with data prediction and the B(h) =
    /// e^{-h} - 1 ("bh2") variant on the VP schedule alpha_t = sqrt(alpha_cumprod), sigma_t = sqrt(1 -
    /// alpha_cumprod), lambda_t = log(alpha_t / sigma_t). The order warms up over the first steps and drops to 1 at
    /// the end (lower_order_final), where the step to sigma = 0 returns the data prediction. UniPC is a
    /// deterministic ODE solver, so <paramref name="eta"/> and <paramref name="noise"/> are not used.
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        if (_lambdas.Length == 0 || _alphaTs.Length == 0 || _sigmaTs.Length == 0)
            throw new InvalidOperationException("Schedule not initialized. Call SetTimesteps() before Step().");

        int stepIndex = FindTimestepIndex(timestep);
        var dataPred = ConvertModelOutputToDataPrediction(modelOutput, timestep, sample);

        // UniC: correct the previous step now that the model has been evaluated at its end point.
        if (_useCorrectorStep && _lastSample is not null && stepIndex > 0 && _dataPredictionHistory.Count > 0)
        {
            sample = UniUpdate(_lastSample, stepIndex - 1, stepIndex, _thisOrder, dataPred);
        }

        _dataPredictionHistory.Add(dataPred);
        _lambdaHistory.Add(_lambdas[stepIndex]);
        if (_dataPredictionHistory.Count > _solverOrder)
        {
            _dataPredictionHistory.RemoveAt(0);
            _lambdaHistory.RemoveAt(0);
        }

        int remainingSteps = Timesteps.Length - stepIndex;
        _thisOrder = Math.Min(Math.Min(_solverOrder, remainingSteps), _lowerOrderNums + 1);
        _thisOrder = Math.Max(1, Math.Min(_thisOrder, _dataPredictionHistory.Count));
        _lastSample = sample;

        // The last step goes to sigma = 0 at first order, where the update is the data prediction itself.
        var prevSample = stepIndex >= Timesteps.Length - 1
            ? dataPred
            : UniUpdate(sample, stepIndex, stepIndex + 1, _thisOrder, null);

        if (_lowerOrderNums < _solverOrder)
            _lowerOrderNums++;
        _stepCounter++;
        return prevSample;
    }

    /// <summary>
    /// The UniP predictor (<paramref name="modelAtTarget"/> null) or UniC corrector update from schedule index
    /// <paramref name="fromIndex"/> to <paramref name="toIndex"/> (diffusers multistep_uni_p_bh_update /
    /// multistep_uni_c_bh_update with predict_x0 and bh2).
    /// </summary>
    private Vector<T> UniUpdate(Vector<T> sample, int fromIndex, int toIndex, int order, Vector<T>? modelAtTarget)
    {
        int last = _dataPredictionHistory.Count - 1;
        var m0 = _dataPredictionHistory[last];
        // The scalar coefficients (h, the phi recurrence, b, the rho solve) are formed in double and each rounded
        // to T once where it multiplies a tensor: the recurrence h_phi_{k+1} = h_phi_k / hh - 1 / (k + 1)! cancels
        // catastrophically for small h, which in Half rounds h_phi_3 to zero and blows b[2] up by two orders.
        double lambdaS0 = NumOps.ToDouble(_lambdas[fromIndex]);
        double lambdaT = NumOps.ToDouble(_lambdas[toIndex]);
        double alphaT = NumOps.ToDouble(_alphaTs[toIndex]);
        double sigmaT = NumOps.ToDouble(_sigmaTs[toIndex]);
        double sigmaS0 = NumOps.ToDouble(_sigmaTs[fromIndex]);
        double h = lambdaT - lambdaS0;
        double hh = -h;
        double hPhi1 = Math.Exp(hh) - 1.0;
        double bh = hPhi1;

        // x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        var result = Engine.Subtract(
            Engine.Multiply(sample, NumOps.FromDouble(sigmaT / sigmaS0)),
            Engine.Multiply(m0, NumOps.FromDouble(alphaT * hPhi1)));

        // b_k = h_phi_k * k! / B(h) for k = 1..3 (h_phi_{k+1} = h_phi_k / hh - 1 / (k + 1)!).
        double hPhiK1 = (hPhi1 / hh) - 1.0;
        double hPhiK2 = (hPhiK1 / hh) - 0.5;
        double hPhiK3 = (hPhiK2 / hh) - (1.0 / 6.0);
        var b = new Vector<double>(new[] { hPhiK1 / bh, 2.0 * hPhiK2 / bh, 6.0 * hPhiK3 / bh });
        double scale = alphaT * bh;

        // Earlier data predictions m_1..m_{order-1} (newest first) and r_k = (lambda_{s_k} - lambda_s0) / h.
        int previous = order - 1;
        var history = new List<Vector<T>>(previous);
        for (int k = 1; k <= previous; k++)
            history.Add(_dataPredictionHistory[last - k]);
        var ratios = HistoryRatios(last, previous, lambdaS0, h);

        Vector<double> rhos;
        if (modelAtTarget is null)
        {
            if (previous == 0)
                return result;
            // Predictor: rho = 0.5 at order 2, otherwise solve R[:-1, :-1] rho = b[:-1].
            rhos = order == 2
                ? SchedulerTensorOps.Full(1, 0.5)
                : SchedulerTensorOps.SolveTransposedVandermonde(ratios, b.GetSubVector(0, previous));
        }
        else
        {
            // Corrector: rho = 0.5 at order 1, otherwise solve R rho = b with the nodes r_1..r_{order-1}, 1.
            rhos = order == 1
                ? SchedulerTensorOps.Full(1, 0.5)
                : SchedulerTensorOps.SolveTransposedVandermonde(
                    Vector<double>.Concatenate(ratios, SchedulerTensorOps.Full(1, 1.0)), b.GetSubVector(0, order));
            // rho_last * (m_t - m0)
            double rhoTarget = rhos[rhos.Length - 1];
            result = Engine.Subtract(result,
                Engine.Multiply(Engine.Subtract(modelAtTarget, m0), NumOps.FromDouble(scale * rhoTarget)));
            if (previous == 0)
                return result;
            rhos = rhos.GetSubVector(0, previous);
        }

        // sum_k rho_k * D1_k with D1_k = (m_k - m0) / r_k.
        var coefficients = DifferenceCoefficients(rhos, ratios, previous, scale, out double weightSum);
        result = SchedulerTensorOps.AddWeightedSum(result, coefficients, history);
        return Engine.Add(result, Engine.Multiply(m0, NumOps.FromDouble(scale * weightSum)));
    }

    /// <summary>
    /// r_k = (lambda_{s_k} - lambda_s0) / h for the <paramref name="previous"/> history nodes, newest first, in double.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("One scalar per history node (at most the solver order minus one), formed in double.")]
    private Vector<double> HistoryRatios(int last, int previous, double lambdaS0, double h)
    {
        var ratios = new double[previous];
        for (int k = 1; k <= previous; k++)
            ratios[k - 1] = (NumOps.ToDouble(_lambdaHistory[last - k]) - lambdaS0) / h;
        return new Vector<double>(ratios);
    }

    /// <summary>
    /// The coefficients -scale * rho_k / r_k rounded to T once each, and the sum of rho_k / r_k.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("One scalar per history node (at most the solver order minus one), rounded from double to T once.")]
    private static Vector<T> DifferenceCoefficients(Vector<double> rhos, Vector<double> ratios, int previous, double scale,
        out double weightSum)
    {
        var coefficients = new T[previous];
        weightSum = 0.0;
        for (int k = 0; k < previous; k++)
        {
            double weight = rhos[k] / ratios[k];
            weightSum += weight;
            coefficients[k] = NumOps.FromDouble(-scale * weight);
        }
        return new Vector<T>(coefficients);
    }

    /// <summary>
    /// Converts model output to data prediction (x_0) form.
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
    /// Computes alpha_t = sqrt(alpha_cumprod), sigma_t = sqrt(1 - alpha_cumprod) and the half log-SNR
    /// lambda_t = log(alpha_t / sigma_t) at each inference timestep.
    /// </summary>
    private void ComputeScheduleValues()
    {
        var alphaCumprod = SchedulerTensorOps.Gather(AlphasCumulativeProduct, Timesteps);
        var oneMinusAlphaCumprod = Engine.Subtract(SchedulerTensorOps.Full(alphaCumprod.Length, NumOps.One), alphaCumprod);
        _alphaTs = Engine.Sqrt(alphaCumprod);
        _sigmaTs = Engine.Sqrt(oneMinusAlphaCumprod);
        var logRatio = Engine.TensorSubtract(Engine.TensorLog(alphaCumprod.AsTensor()), Engine.TensorLog(oneMinusAlphaCumprod.AsTensor()));
        _lambdas = Engine.TensorMultiplyScalar(logRatio, NumOps.FromDouble(0.5)).ToVector();
    }

    /// <summary>
    /// Resets the scheduler state for a new generation run.
    /// </summary>
    private void ResetState()
    {
        _dataPredictionHistory.Clear();
        _lambdaHistory.Clear();
        _lastSample = null;
        _thisOrder = 1;
        _lowerOrderNums = 0;
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
