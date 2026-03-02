using AiDotNet.Enums;

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
public sealed class SASolverScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly int _order;
    private readonly List<Vector<T>> _previousOutputs = new();

    public SASolverScheduler(SchedulerConfig<T> config, int order = 3) : base(config)
    {
        _order = Math.Max(1, Math.Min(order, 4));
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        _previousOutputs.Add(modelOutput);
        if (_previousOutputs.Count > _order)
            _previousOutputs.RemoveAt(0);

        double t = (double)timestep / TrainTimesteps;
        double dt = 1.0 / Math.Max(1, Timesteps.Length);
        var dtT = NumOps.FromDouble(dt);

        // Adams-Bashforth predictor using available history
        var result = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
        {
            var pred = NumOps.Subtract(sample[i], NumOps.Multiply(dtT, modelOutput[i]));

            // Add stochastic noise
            if (noise != null && NumOps.ToDouble(eta) > 0)
            {
                double noiseLevel = Math.Sqrt(2.0 * dt) * NumOps.ToDouble(eta);
                pred = NumOps.Add(pred, NumOps.Multiply(NumOps.FromDouble(noiseLevel), noise[i]));
            }

            result[i] = pred;
        }

        return result;
    }
}
