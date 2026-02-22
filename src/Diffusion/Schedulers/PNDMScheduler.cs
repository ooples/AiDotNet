namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// PNDM (Pseudo Numerical Methods for Diffusion Models) scheduler implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PNDM uses pseudo numerical methods to accelerate diffusion sampling. It can achieve
/// high-quality results with even fewer steps than DDIM by using a combination of
/// linear multi-step methods and improved transfer techniques.
/// </para>
/// <para>
/// <b>For Beginners:</b> PNDM is an advanced method for fast image generation.
///
/// Think of diffusion like walking down a mountain (from noise to clean image):
/// - DDPM: Take 1000 tiny careful steps
/// - DDIM: Take 50 medium steps
/// - PNDM: Take 20 smart steps using "momentum" from previous steps
///
/// The key insight is that PNDM remembers its previous predictions and uses them
/// to make better guesses about where to step next. It's like a ball rolling down
/// a hill - it uses its momentum to move faster.
///
/// Advantages:
/// - Very fast generation (often 20-25 steps for good quality)
/// - Uses multi-step methods from numerical analysis
/// - Good balance of speed and quality
///
/// The scheduler operates in two phases:
/// 1. Prk (Pseudo Runge-Kutta) for initial steps
/// 2. Plms (Pseudo Linear Multi-Step) for remaining steps
/// </para>
/// <para>
/// <b>Reference:</b> "Pseudo Numerical Methods for Diffusion Models on Manifolds" by Liu et al., 2022
/// </para>
/// </remarks>
public sealed class PNDMScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// History of model outputs for multi-step methods.
    /// </summary>
    private readonly List<Vector<T>> _ets = new();

    /// <summary>
    /// Current step counter within a single inference run.
    /// </summary>
    private int _counter;

    /// <summary>
    /// Current sample being processed (for Runge-Kutta steps).
    /// </summary>
    private Vector<T>? _currentSample;

    /// <summary>
    /// Number of warmup steps before switching to linear multi-step.
    /// </summary>
    private const int PrkModeSteps = 4;

    /// <summary>
    /// Initializes a new instance of the PNDM scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a PNDM scheduler for fast, high-quality generation.
    /// PNDM works best with 20-50 inference steps.
    /// </para>
    /// <example>
    /// <code>
    /// // Create PNDM scheduler with default settings
    /// var config = SchedulerConfig&lt;double&gt;.CreateDefault();
    /// var scheduler = new PNDMScheduler&lt;double&gt;(config);
    ///
    /// // Set up for 25 inference steps (PNDM works well with few steps)
    /// scheduler.SetTimesteps(25);
    /// </code>
    /// </example>
    /// </remarks>
    public PNDMScheduler(SchedulerConfig<T> config) : base(config)
    {
        _counter = 0;
    }

    /// <summary>
    /// Sets up the inference timesteps and resets the scheduler state.
    /// </summary>
    /// <param name="inferenceSteps">Number of denoising steps to use during inference.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when inferenceSteps is invalid.</exception>
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        ResetState();
    }

    /// <summary>
    /// Resets the scheduler state for a new generation run.
    /// </summary>
    private void ResetState()
    {
        _ets.Clear();
        _counter = 0;
        _currentSample = null;
    }

    /// <summary>
    /// Performs one PNDM denoising step.
    /// </summary>
    /// <param name="modelOutput">The model's noise prediction (epsilon).</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">Not used in PNDM (deterministic scheduler). Included for interface compatibility.</param>
    /// <param name="noise">Not used in PNDM. Included for interface compatibility.</param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <exception cref="ArgumentNullException">Thrown when modelOutput or sample is null.</exception>
    /// <exception cref="ArgumentException">Thrown when modelOutput and sample have different lengths.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when timestep is out of range.</exception>
    /// <remarks>
    /// <para>
    /// PNDM uses a two-phase approach:
    /// 1. First few steps use pseudo Runge-Kutta (prk) method
    /// 2. Remaining steps use pseudo linear multi-step (plms) method
    ///
    /// The plms method uses history of previous predictions to extrapolate
    /// a better estimate for the current step.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method intelligently combines current and past
    /// predictions to take larger, more accurate steps. It's like using a running
    /// average to smooth out predictions and make faster progress.
    /// </para>
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        // Ensure SetTimesteps was called before Step
        if (Timesteps.Length == 0)
            throw new InvalidOperationException("Timesteps not initialized. Call SetTimesteps() before Step().");

        // Use PRK mode for warmup (first 4 steps), then switch to PLMS mode
        var result = _counter < PrkModeSteps
            ? StepPrk(modelOutput, timestep, sample)
            : StepPlms(modelOutput, timestep, sample);

        _counter++;
        return result;
    }

    /// <summary>
    /// Performs a pseudo Runge-Kutta step (warmup phase).
    /// </summary>
    private Vector<T> StepPrk(Vector<T> modelOutput, int timestep, Vector<T> sample)
    {
        // Store current sample if this is a new timestep
        _currentSample ??= CopyVector(sample);

        int diffToPrev = (Config.TrainTimesteps / Timesteps.Length) / 2;
        int prevTimestep = Math.Max(timestep - diffToPrev, 0);

        // Get alpha cumulative product for previous timestep
        T alphaCumprodPrev = AlphasCumulativeProduct[prevTimestep];

        // Compute prediction based on counter within prk phase
        Vector<T> predOriginalSample;
        Vector<T> result;

        switch (_counter % 4)
        {
            case 0:
                // First prk step: predict x_0 and store
                predOriginalSample = PredictOriginalSample(modelOutput, timestep, sample);
                predOriginalSample = ClipSampleIfNeeded(predOriginalSample);

                // Store model output for linear combination
                _ets.Add(CopyVector(modelOutput));

                // Compute intermediate sample
                result = ComputePrevSample(predOriginalSample, modelOutput, alphaCumprodPrev);
                break;

            case 1:
                // Second prk step
                predOriginalSample = PredictOriginalSample(modelOutput, timestep, sample);
                predOriginalSample = ClipSampleIfNeeded(predOriginalSample);

                // Update ets with average (VECTORIZED)
                var sum = Engine.Add(_ets[^1], modelOutput);
                var avgEt = Engine.Multiply(sum, NumOps.FromDouble(0.5));
                _ets[^1] = avgEt;

                result = ComputePrevSample(predOriginalSample, avgEt, alphaCumprodPrev);
                break;

            case 2:
                // Third prk step
                predOriginalSample = PredictOriginalSample(modelOutput, timestep, sample);
                predOriginalSample = ClipSampleIfNeeded(predOriginalSample);

                _ets.Add(CopyVector(modelOutput));
                result = ComputePrevSample(predOriginalSample, modelOutput, alphaCumprodPrev);
                break;

            case 3:
            default:
                // Fourth prk step: compute linear combination (VECTORIZED)
                var oneThird = NumOps.FromDouble(1.0 / 3.0);

                // term1 = (1/3) * _ets[^2]
                var term1 = Engine.Multiply(_ets[^2], oneThird);
                // term2 = (2/3) * ((_ets[^1] + modelOutput) / 2) = (1/3) * (_ets[^1] + modelOutput)
                var avgLast = Engine.Add(_ets[^1], modelOutput);
                var term2 = Engine.Multiply(avgLast, oneThird);
                var linearCombination = Engine.Add(term1, term2);

                // Update last ets entry
                _ets[^1] = linearCombination;

                // Final computation for this timestep
                int actualPrevTimestep = Math.Max(timestep - (Config.TrainTimesteps / Timesteps.Length), 0);
                T alphaCumprodActualPrev = AlphasCumulativeProduct[actualPrevTimestep];

                // Validate _currentSample is set before using
                if (_currentSample == null)
                {
                    throw new InvalidOperationException("Current sample not initialized for PRK step.");
                }
                predOriginalSample = PredictOriginalSample(linearCombination, timestep, _currentSample);
                predOriginalSample = ClipSampleIfNeeded(predOriginalSample);

                result = ComputePrevSample(predOriginalSample, linearCombination, alphaCumprodActualPrev);
                _currentSample = null;
                break;
        }

        return result;
    }

    /// <summary>
    /// Performs a pseudo linear multi-step (plms) step.
    /// </summary>
    private Vector<T> StepPlms(Vector<T> modelOutput, int timestep, Vector<T> sample)
    {
        // Store model output
        _ets.Add(CopyVector(modelOutput));

        // Keep only the last 4 outputs for memory efficiency
        while (_ets.Count > 4)
        {
            _ets.RemoveAt(0);
        }

        // Compute linear combination based on available history
        var etPrime = ComputeLinearMultiStep();

        int prevTimestep = Math.Max(timestep - (Config.TrainTimesteps / Timesteps.Length), 0);
        T alphaCumprodPrev = AlphasCumulativeProduct[prevTimestep];

        var predOriginalSample = PredictOriginalSample(etPrime, timestep, sample);
        predOriginalSample = ClipSampleIfNeeded(predOriginalSample);

        return ComputePrevSample(predOriginalSample, etPrime, alphaCumprodPrev);
    }

    /// <summary>
    /// Computes the linear multi-step combination of model outputs (VECTORIZED).
    /// Uses Adams-Bashforth style coefficients for multi-step prediction.
    /// </summary>
    private Vector<T> ComputeLinearMultiStep()
    {
        // Adams-Bashforth style coefficients with vectorized operations
        switch (_ets.Count)
        {
            case 1:
                // Simple copy - return the only element
                return CopyVector(_ets[0]);

            case 2:
                // Linear extrapolation: (3 * e_{t-1} - e_{t-2}) / 2
                var half = NumOps.FromDouble(0.5);
                var three = NumOps.FromDouble(3.0);
                var scaled1 = Engine.Multiply(_ets[1], three);
                var diff = Engine.Subtract(scaled1, _ets[0]);
                return Engine.Multiply(diff, half);

            case 3:
                // Quadratic: (23 * e_{t-1} - 16 * e_{t-2} + 5 * e_{t-3}) / 12
                var c1 = NumOps.FromDouble(23.0 / 12.0);
                var c2 = NumOps.FromDouble(-16.0 / 12.0);
                var c3 = NumOps.FromDouble(5.0 / 12.0);
                var term1 = Engine.Multiply(_ets[2], c1);
                var term2 = Engine.Multiply(_ets[1], c2);
                var term3 = Engine.Multiply(_ets[0], c3);
                var sum12 = Engine.Add(term1, term2);
                return Engine.Add(sum12, term3);

            default:
                // Cubic: (55 * e_{t-1} - 59 * e_{t-2} + 37 * e_{t-3} - 9 * e_{t-4}) / 24
                var d1 = NumOps.FromDouble(55.0 / 24.0);
                var d2 = NumOps.FromDouble(-59.0 / 24.0);
                var d3 = NumOps.FromDouble(37.0 / 24.0);
                var d4 = NumOps.FromDouble(-9.0 / 24.0);
                var t1 = Engine.Multiply(_ets[3], d1);
                var t2 = Engine.Multiply(_ets[2], d2);
                var t3 = Engine.Multiply(_ets[1], d3);
                var t4 = Engine.Multiply(_ets[0], d4);
                var s12 = Engine.Add(t1, t2);
                var s34 = Engine.Add(t3, t4);
                return Engine.Add(s12, s34);
        }
    }

    /// <summary>
    /// Predicts the original sample from noise prediction (VECTORIZED).
    /// </summary>
    private Vector<T> PredictOriginalSample(Vector<T> modelOutput, int timestep, Vector<T> sample)
    {
        T alphaCumprod = AlphasCumulativeProduct[timestep];
        T sqrtAlphaCumprod = NumOps.Sqrt(alphaCumprod);
        T sqrtOneMinusAlphaCumprod = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));

        // Vectorized: x_0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
        var noiseTerm = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprod);
        var numerator = Engine.Subtract(sample, noiseTerm);
        return Engine.Divide(numerator, sqrtAlphaCumprod);
    }

    /// <summary>
    /// Computes the previous sample from predicted original and model output (VECTORIZED).
    /// </summary>
    private Vector<T> ComputePrevSample(Vector<T> predOriginal, Vector<T> modelOutput, T alphaCumprodPrev)
    {
        T sqrtAlphaCumprodPrev = NumOps.Sqrt(alphaCumprodPrev);
        T sqrtOneMinusAlphaCumprodPrev = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprodPrev));

        // Vectorized: x_{t-1} = sqrt(alpha_prev) * x_0 + sqrt(1-alpha_prev) * eps
        var originalTerm = Engine.Multiply(predOriginal, sqrtAlphaCumprodPrev);
        var noiseTerm = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprodPrev);
        return Engine.Add(originalTerm, noiseTerm);
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
        state["counter"] = _counter;
        state["ets_count"] = _ets.Count;
        return state;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Note: The counter is not restored because the model output history (_ets) cannot be
    /// serialized without significant overhead. Restoring counter without history would cause
    /// incorrect behavior in PLMS mode which relies on previous outputs.
    /// After loading state, the scheduler will restart from the warmup (PRK) phase.
    /// </remarks>
    public override void LoadState(Dictionary<string, object> state)
    {
        base.LoadState(state);
        ResetState();
        // Counter is intentionally not restored - see remarks above
    }
}
