using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Schedulers;

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
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public sealed class DPMSolverMultistepScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// History of model outputs for multi-step methods.
    /// </summary>
    private readonly List<Vector<T>> _modelOutputHistory = new();

    /// <summary>
    /// Lambda values (log-SNR) for each inference timestep.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private Vector<T>? _lambdas;

    /// <summary>
    /// Alpha_t values for inference timesteps.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private Vector<T>? _alphaTs;

    /// <summary>
    /// Sigma_t values for inference timesteps.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
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
        // diffusers DPMSolverMultistepScheduler (dpmsolver++): alpha_t = sqrt(alpha_cumprod),
        // sigma_t = sqrt(1 - alpha_cumprod), lambda_t = log(alpha_t) - log(sigma_t).
        var timesteps = Timesteps;
        var index = new Tensor<int>(new[] { timesteps.Length }, new Vector<int>(timesteps));
        var alphaCumprod = Engine.TensorIndexSelect(AlphasCumulativeProduct.AsTensor(), index, 0).ToVector();

        _alphaTs = Engine.Sqrt(alphaCumprod);
        _sigmaTs = Engine.Sqrt(Engine.Subtract(Vector<T>.Ones(timesteps.Length), alphaCumprod));
        _lambdas = Engine.Subtract(Engine.Log(_alphaTs), Engine.Log(_sigmaTs));
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

        // Determine effective order (can't use higher order than available history). Like diffusers'
        // lower_order_final, short schedules (< 15 steps) take at most a second-order step second to last.
        int effectiveOrder = Math.Min(_stepCounter + 1, _solverOrder);
        effectiveOrder = Math.Min(effectiveOrder, _modelOutputHistory.Count);
        if (stepIndex == Timesteps.Length - 2 && Timesteps.Length < 15)
            effectiveOrder = Math.Min(effectiveOrder, 2);

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
        var (lambdas, alphaTs, sigmaTs) = GetSchedule();

        T alpha_t = alphaTs[nextStepIndex];
        T sigma_t = sigmaTs[nextStepIndex];
        T sigma_s = sigmaTs[stepIndex];
        T h = NumOps.Subtract(lambdas[nextStepIndex], lambdas[stepIndex]);

        // DPM-Solver++ first order: x_t = (sigma_t / sigma_s) x_s - alpha_t (exp(-h) - 1) D_0
        T phi = NumOps.Multiply(alpha_t, NumOps.Subtract(NumOps.Exp(NumOps.Negate(h)), NumOps.One));

        var scaledSample = Engine.Multiply(sample, NumOps.Divide(sigma_t, sigma_s));
        return Engine.Subtract(scaledSample, Engine.Multiply(_modelOutputHistory[^1], phi));
    }

    /// <summary>
    /// Second-order multistep DPM-Solver++ update (diffusers' midpoint solver type).
    /// </summary>
    private Vector<T> SecondOrderUpdate(Vector<T> sample, int stepIndex, int nextStepIndex)
    {
        if (_modelOutputHistory.Count < 2 || stepIndex < 1)
            return FirstOrderUpdate(sample, stepIndex, nextStepIndex);

        var (lambdas, alphaTs, sigmaTs) = GetSchedule();

        T alpha_t = alphaTs[nextStepIndex];
        T sigma_t = sigmaTs[nextStepIndex];
        T sigma_s0 = sigmaTs[stepIndex];
        T h = NumOps.Subtract(lambdas[nextStepIndex], lambdas[stepIndex]);
        T h0 = NumOps.Subtract(lambdas[stepIndex], lambdas[stepIndex - 1]);
        T r0 = NumOps.Divide(h0, h);

        var m0 = _modelOutputHistory[^1];
        var m1 = _modelOutputHistory[^2];

        // D0 = m0, D1 = (m0 - m1) / r0
        // x_t = (sigma_t / sigma_s0) x - alpha_t (exp(-h) - 1) D0 - 0.5 alpha_t (exp(-h) - 1) D1
        var d1 = Engine.Divide(Engine.Subtract(m0, m1), r0);
        T phi = NumOps.Multiply(alpha_t, NumOps.Subtract(NumOps.Exp(NumOps.Negate(h)), NumOps.One));

        var result = Engine.Multiply(sample, NumOps.Divide(sigma_t, sigma_s0));
        result = Engine.Subtract(result, Engine.Multiply(m0, phi));
        return Engine.Subtract(result, Engine.Multiply(d1, NumOps.Multiply(NumOps.FromDouble(0.5), phi)));
    }

    /// <summary>
    /// Third-order multistep DPM-Solver++ update.
    /// </summary>
    private Vector<T> ThirdOrderUpdate(Vector<T> sample, int stepIndex, int nextStepIndex)
    {
        if (_modelOutputHistory.Count < 3 || stepIndex < 2)
            return SecondOrderUpdate(sample, stepIndex, nextStepIndex);

        var (lambdas, alphaTs, sigmaTs) = GetSchedule();

        T alpha_t = alphaTs[nextStepIndex];
        T sigma_t = sigmaTs[nextStepIndex];
        T sigma_s0 = sigmaTs[stepIndex];
        T h = NumOps.Subtract(lambdas[nextStepIndex], lambdas[stepIndex]);
        T h0 = NumOps.Subtract(lambdas[stepIndex], lambdas[stepIndex - 1]);
        T h1 = NumOps.Subtract(lambdas[stepIndex - 1], lambdas[stepIndex - 2]);
        T r0 = NumOps.Divide(h0, h);
        T r1 = NumOps.Divide(h1, h);
        T r0PlusR1 = NumOps.Add(r0, r1);

        var m0 = _modelOutputHistory[^1];
        var m1 = _modelOutputHistory[^2];
        var m2 = _modelOutputHistory[^3];

        // D1_0 = (m0 - m1) / r0, D1_1 = (m1 - m2) / r1
        // D1 = D1_0 + r0 / (r0 + r1) (D1_0 - D1_1), D2 = (D1_0 - D1_1) / (r0 + r1)
        var d10 = Engine.Divide(Engine.Subtract(m0, m1), r0);
        var d11 = Engine.Divide(Engine.Subtract(m1, m2), r1);
        var d1Diff = Engine.Subtract(d10, d11);
        var d1 = Engine.Add(d10, Engine.Multiply(d1Diff, NumOps.Divide(r0, r0PlusR1)));
        var d2 = Engine.Divide(d1Diff, r0PlusR1);

        // x_t = (sigma_t / sigma_s0) x - alpha_t (exp(-h) - 1) D0
        //       + alpha_t ((exp(-h) - 1) / h + 1) D1 - alpha_t ((exp(-h) - 1 + h) / h^2 - 0.5) D2
        T expM1 = NumOps.Subtract(NumOps.Exp(NumOps.Negate(h)), NumOps.One);
        T coeff0 = NumOps.Multiply(alpha_t, expM1);
        T coeff1 = NumOps.Multiply(alpha_t, NumOps.Add(NumOps.Divide(expM1, h), NumOps.One));
        T coeff2 = NumOps.Multiply(alpha_t,
            NumOps.Subtract(NumOps.Divide(NumOps.Add(expM1, h), NumOps.Multiply(h, h)), NumOps.FromDouble(0.5)));

        var result = Engine.Multiply(sample, NumOps.Divide(sigma_t, sigma_s0));
        result = Engine.Subtract(result, Engine.Multiply(m0, coeff0));
        result = Engine.Add(result, Engine.Multiply(d1, coeff1));
        return Engine.Subtract(result, Engine.Multiply(d2, coeff2));
    }

    private (Vector<T> Lambdas, Vector<T> AlphaTs, Vector<T> SigmaTs) GetSchedule()
    {
        var lambdas = _lambdas
            ?? throw new InvalidOperationException("Lambda values have not been initialized. Call SetTimesteps() first.");
        var alphaTs = _alphaTs
            ?? throw new InvalidOperationException("Alpha values have not been initialized. Call SetTimesteps() first.");
        var sigmaTs = _sigmaTs
            ?? throw new InvalidOperationException("Sigma values have not been initialized. Call SetTimesteps() first.");
        return (lambdas, alphaTs, sigmaTs);
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
    private static Vector<T> CopyVector(Vector<T> source) => source.Clone();

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
