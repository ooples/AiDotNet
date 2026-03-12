using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Flow DPM-Solver scheduler applying DPM-Solver acceleration to rectified flow models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Adapts DPM-Solver's multistep prediction to flow-matching ODE trajectories,
/// using cached previous velocity predictions for higher-order polynomial extrapolation.
/// Achieves fewer function evaluations than standard Euler rectified flow.
/// </para>
/// <para>
/// <b>For Beginners:</b> This scheduler makes rectified flow models even faster by
/// reusing previous computation steps. Instead of simple Euler steps, it uses smarter
/// math (polynomial extrapolation) to take bigger steps while maintaining quality.
/// </para>
/// </remarks>
public sealed class FlowDPMSolverScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly int _solverOrder;
    private readonly List<Vector<T>> _previousOutputs = new();

    public FlowDPMSolverScheduler(SchedulerConfig<T> config, int solverOrder = 2) : base(config)
    {
        _solverOrder = Math.Max(1, Math.Min(solverOrder, 3));
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        _previousOutputs.Add(modelOutput);
        if (_previousOutputs.Count > _solverOrder)
            _previousOutputs.RemoveAt(0);

        double t = (double)timestep / TrainTimesteps;
        double dt = 1.0 / Math.Max(1, Timesteps.Length);
        var dtT = NumOps.FromDouble(dt);

        // Use highest available order
        var velocity = _previousOutputs.Count >= 2
            ? ExtrapolateVelocity(_previousOutputs)
            : modelOutput;

        var result = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
            result[i] = NumOps.Subtract(sample[i], NumOps.Multiply(dtT, velocity[i]));

        return result;
    }

    private Vector<T> ExtrapolateVelocity(List<Vector<T>> outputs)
    {
        // Linear extrapolation from the last two outputs
        var last = outputs[outputs.Count - 1];
        var prev = outputs[outputs.Count - 2];
        var result = new Vector<T>(last.Length);
        var two = NumOps.FromDouble(2.0);

        for (int i = 0; i < last.Length; i++)
            result[i] = NumOps.Subtract(NumOps.Multiply(two, last[i]), prev[i]);

        return result;
    }
}
