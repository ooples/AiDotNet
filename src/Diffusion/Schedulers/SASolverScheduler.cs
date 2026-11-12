using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// SA-Solver (Stochastic Adams) scheduler using Adams-Bashforth/Moulton methods for SDE sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SA-Solver applies Adams-Bashforth (predictor) and Adams-Moulton (corrector) multistep
/// methods to stochastic differential equation sampling. This enables efficient SDE sampling
/// with higher-order accuracy using cached function evaluations.
/// </para>
/// <para>
/// <b>For Beginners:</b> SA-Solver is a multistep sampler that reuses previous computations
/// (like DPM-Solver) but for stochastic (random) sampling. This gives you the diversity
/// benefits of SDE sampling with the efficiency of multistep methods.
/// </para>
/// <para>
/// Reference: Xue et al., "SA-Solver: Stochastic Adams Solver for Fast Training of Diffusion Models", NeurIPS 2023
/// </para>
/// </remarks>
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public sealed class SASolverScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly int _order;
    private readonly List<Vector<T>> _dataPredictions = new();
    private readonly List<T> _lambdas = new();
    private Vector<T>? _lastSample;
    private Vector<T>? _lastNoise;
    private T _lastLambda;
    private T _lastSigma;
    private T _lastTau;

    public SASolverScheduler(SchedulerConfig<T> config, int order = 3) : base(config)
    {
        _order = Math.Max(1, Math.Min(order, 4));
        _lastLambda = NumOps.Zero;
        _lastSigma = NumOps.One;
        _lastTau = NumOps.Zero;
    }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        // A new trajectory must not reuse the multistep history of the previous one.
        _dataPredictions.Clear();
        _lambdas.Clear();
        _lastSample = null;
        _lastNoise = null;
    }

    /// <inheritdoc />
    /// <remarks>
    /// SA-Solver (Xue et al., NeurIPS 2023) in data-prediction form on the VP schedule
    /// (alpha_t = sqrt(alpha_cumprod), sigma_t = sqrt(1 - alpha_cumprod), lambda_t = log(alpha_t / sigma_t)).
    /// Each step first applies the stochastic Adams-Moulton corrector to the previous step, now that the data
    /// prediction at its end point is known, then the stochastic Adams-Bashforth predictor:
    /// x_t = (sigma_t / sigma_s) e^{-tau^2 h} x_s + alpha_t sum_i b_i x0_i + sigma_t sqrt(1 - e^{-2 tau^2 h}) z,
    /// where b_i integrates (1 + tau^2) e^{-(1 + tau^2)(lambda_t - lambda)} times the Lagrange basis of the history
    /// over [lambda_s, lambda_t]. As in the diffusers default tau function, tau = eta (when noise is supplied) for
    /// timesteps in [0.2 T, 0.8 T] and 0 elsewhere; the final step to sigma = 0 returns the data prediction.
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);
        var timesteps = Timesteps;
        int index = Array.IndexOf(timesteps, timestep);

        T alphaCumprod = AlphasCumulativeProduct[timestep];
        T alphaS = NumOps.Sqrt(alphaCumprod);
        T sigmaS = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));
        T lambdaS = NumOps.Subtract(NumOps.Log(alphaS), NumOps.Log(sigmaS));
        var dataPrediction = ClipSampleIfNeeded(ToDataPrediction(modelOutput, sample, alphaS, sigmaS));

        // Corrector for the previous step, integrating over the history plus the new data prediction.
        if (_lastSample is not null && _dataPredictions.Count > 0)
        {
            int correctorOrder = Math.Min(_order, _dataPredictions.Count + 1);
            int start = _dataPredictions.Count - correctorOrder + 1;
            var values = _dataPredictions.GetRange(start, correctorOrder - 1);
            values.Add(dataPrediction);
            var lambdas = _lambdas.GetRange(start, correctorOrder - 1);
            lambdas.Add(lambdaS);
            sample = StochasticAdams(_lastSample, _lastLambda, _lastSigma, lambdaS, alphaS, sigmaS, _lastTau,
                lambdas, values, _lastNoise);
        }

        _dataPredictions.Add(dataPrediction);
        _lambdas.Add(lambdaS);
        if (_dataPredictions.Count > _order)
        {
            _dataPredictions.RemoveAt(0);
            _lambdas.RemoveAt(0);
        }

        int previousTimestep = index >= 0
            ? (index + 1 < timesteps.Length ? timesteps[index + 1] : -1)
            : timestep - TrainTimesteps / Math.Max(1, timesteps.Length);
        if (previousTimestep < 0)
        {
            // sigma_t = 0: the solution is the data prediction itself.
            _lastSample = null;
            _lastNoise = null;
            return dataPrediction;
        }

        T alphaCumprodNext = AlphasCumulativeProduct[previousTimestep];
        T alphaT = NumOps.Sqrt(alphaCumprodNext);
        T sigmaT = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprodNext));
        T lambdaT = NumOps.Subtract(NumOps.Log(alphaT), NumOps.Log(sigmaT));
        bool stochastic = noise is not null && timestep >= TrainTimesteps / 5 && timestep <= 4 * TrainTimesteps / 5;
        T tau = stochastic ? eta : NumOps.Zero;
        var stepNoise = stochastic ? noise : null;

        int predictorOrder = Math.Min(_order, _dataPredictions.Count);
        var result = StochasticAdams(sample, lambdaS, sigmaS, lambdaT, alphaT, sigmaT, tau,
            _lambdas.GetRange(_lambdas.Count - predictorOrder, predictorOrder),
            _dataPredictions.GetRange(_dataPredictions.Count - predictorOrder, predictorOrder),
            stepNoise);

        _lastSample = sample;
        _lastNoise = stepNoise;
        _lastLambda = lambdaS;
        _lastSigma = sigmaS;
        _lastTau = tau;
        return result;
    }

    /// <summary>
    /// One stochastic Adams step from (lambda_s, x_s) to lambda_t using the data predictions at the given nodes.
    /// </summary>
    private Vector<T> StochasticAdams(Vector<T> sample, T lambdaFrom, T sigmaFrom, T lambdaTo, T alphaTo, T sigmaTo,
        T tau, List<T> nodeLambdas, List<Vector<T>> nodeValues, Vector<T>? noise)
    {
        T tauSquared = NumOps.Multiply(tau, tau);
        T h = NumOps.Subtract(lambdaTo, lambdaFrom);
        T tauSquaredH = NumOps.Multiply(tauSquared, h);
        T decay = NumOps.Exp(NumOps.Negate(tauSquaredH));
        var result = Engine.Multiply(sample, NumOps.Multiply(NumOps.Divide(sigmaTo, sigmaFrom), decay));

        var offsets = Engine.Subtract(new Vector<T>(nodeLambdas), SchedulerTensorOps.Full(nodeLambdas.Count, lambdaTo));
        var coefficients = SchedulerTensorOps.ExponentialAdamsCoefficients(offsets, h, NumOps.Add(NumOps.One, tauSquared));
        result = SchedulerTensorOps.AddWeightedSum(result, Engine.Multiply(coefficients, alphaTo), nodeValues);

        if (noise is null || !NumOps.GreaterThan(tau, NumOps.Zero))
            return result;
        T twoTauSquaredH = NumOps.Multiply(NumOps.FromDouble(2.0), tauSquaredH);
        T noiseScale = NumOps.Multiply(sigmaTo, NumOps.Sqrt(NumOps.Subtract(NumOps.One, NumOps.Exp(NumOps.Negate(twoTauSquaredH)))));
        return Engine.Add(result, Engine.Multiply(noise, noiseScale));
    }

    /// <summary>
    /// Converts the model output to the predicted clean sample x0 for the configured prediction type.
    /// </summary>
    private Vector<T> ToDataPrediction(Vector<T> modelOutput, Vector<T> sample, T alpha, T sigma)
    {
        switch (Config.PredictionType)
        {
            case DiffusionPredictionType.Sample:
                return modelOutput;
            case DiffusionPredictionType.VPrediction:
                return Engine.Subtract(Engine.Multiply(sample, alpha), Engine.Multiply(modelOutput, sigma));
            default:
                return Engine.Divide(Engine.Subtract(sample, Engine.Multiply(modelOutput, sigma)), alpha);
        }
    }
}
