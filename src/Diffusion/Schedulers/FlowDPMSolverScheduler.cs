using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Flow DPM-Solver scheduler applying DPM-Solver acceleration to rectified flow models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Adapts DPM-Solver's multistep prediction to flow-matching ODE trajectories,
/// reusing the data predictions of previous steps in the DPM-Solver++ multistep (2M/3M) updates.
/// Achieves fewer function evaluations than standard Euler rectified flow.
/// </para>
/// <para>
/// <b>For Beginners:</b> This scheduler makes rectified flow models even faster by
/// reusing previous computation steps. Instead of simple Euler steps, it uses smarter
/// math (polynomial extrapolation) to take bigger steps while maintaining quality.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public sealed class FlowDPMSolverScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly int _solverOrder;
    private readonly List<Vector<T>> _dataPredictions = new();
    private readonly List<double> _lambdas = new();
    private int _lowerOrderNums;

    public FlowDPMSolverScheduler(SchedulerConfig<T> config, int solverOrder = 2) : base(config)
    {
        _solverOrder = Math.Max(1, Math.Min(solverOrder, 3));
    }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        // A new trajectory must not reuse the multistep history of the previous one.
        _dataPredictions.Clear();
        _lambdas.Clear();
        _lowerOrderNums = 0;
    }

    /// <inheritdoc />
    /// <remarks>
    /// DPM-Solver++ multistep (Lu et al., 2022) on the flow-matching path x_t = (1 - t) x_0 + t * eps, as in
    /// diffusers' DPMSolverMultistepScheduler with flow sigmas: alpha_t = 1 - t, sigma_t = t,
    /// lambda_t = log(alpha_t / sigma_t), the velocity v is turned into the data prediction x_0 = x_t - t * v,
    /// and the 1st/2nd/3rd-order updates use the midpoint 2M and 3M formulas. The first steps warm up at lower
    /// order and the final step (to t = 0) is first order (diffusers' lower_order_final, plus
    /// lower_order_second for schedules shorter than 15 steps).
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        var timesteps = Timesteps;
        int index = Array.IndexOf(timesteps, timestep);
        double s = SchedulerTensorOps.FlowTime(timestep, TrainTimesteps);
        double sNext = index >= 0
            ? (index + 1 < timesteps.Length ? SchedulerTensorOps.FlowTime(timesteps[index + 1], TrainTimesteps) : 0.0)
            : Math.Max(0.0, s - 1.0 / Math.Max(1, timesteps.Length));

        var dataPrediction = Engine.Subtract(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(s)));
        double lambdaS = Lambda(s);
        _dataPredictions.Add(dataPrediction);
        _lambdas.Add(lambdaS);
        if (_dataPredictions.Count > _solverOrder)
        {
            _dataPredictions.RemoveAt(0);
            _lambdas.RemoveAt(0);
        }

        bool lowerOrderFinal = index < 0 || index == timesteps.Length - 1;
        bool lowerOrderSecond = index >= 0 && index == timesteps.Length - 2 && timesteps.Length < 15;
        int order;
        if (_solverOrder == 1 || _lowerOrderNums < 1 || lowerOrderFinal)
            order = 1;
        else if (_solverOrder == 2 || _lowerOrderNums < 2 || lowerOrderSecond)
            order = 2;
        else
            order = 3;
        if (_lowerOrderNums < _solverOrder)
            _lowerOrderNums++;

        double alphaT = 1.0 - sNext;
        double sigmaT = sNext;
        double h = Lambda(sNext) - lambdaS;
        double phi = Math.Exp(-h) - 1.0;
        int last = _dataPredictions.Count - 1;
        var m0 = _dataPredictions[last];

        // First-order (DDIM) part: (sigma_t / sigma_s) x - alpha_t (e^{-h} - 1) D0
        var result = Engine.Add(
            Engine.Multiply(sample, NumOps.FromDouble(s > 0.0 ? sigmaT / s : 0.0)),
            Engine.Multiply(m0, NumOps.FromDouble(-alphaT * phi)));
        if (order == 1)
            return result;

        var m1 = _dataPredictions[last - 1];
        double r0 = (lambdaS - _lambdas[last - 1]) / h;
        var d10 = Engine.Multiply(Engine.Subtract(m0, m1), NumOps.FromDouble(1.0 / r0));
        if (order == 2)
            return Engine.Add(result, Engine.Multiply(d10, NumOps.FromDouble(-0.5 * alphaT * phi)));

        var m2 = _dataPredictions[last - 2];
        double r1 = (_lambdas[last - 1] - _lambdas[last - 2]) / h;
        var d11 = Engine.Multiply(Engine.Subtract(m1, m2), NumOps.FromDouble(1.0 / r1));
        var d1Diff = Engine.Subtract(d10, d11);
        var d1 = Engine.Add(d10, Engine.Multiply(d1Diff, NumOps.FromDouble(r0 / (r0 + r1))));
        var d2 = Engine.Multiply(d1Diff, NumOps.FromDouble(1.0 / (r0 + r1)));
        result = Engine.Add(result, Engine.Multiply(d1, NumOps.FromDouble(alphaT * (phi / h + 1.0))));
        return Engine.Add(result, Engine.Multiply(d2, NumOps.FromDouble(-alphaT * ((phi + h) / (h * h) - 0.5))));
    }

    /// <summary>
    /// The half log-SNR lambda_t = log(alpha_t / sigma_t) of flow time t (alpha_t = 1 - t, sigma_t = t).
    /// </summary>
    private static double Lambda(double t) => Math.Log(1.0 - t) - Math.Log(t);
}
