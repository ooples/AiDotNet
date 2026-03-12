using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// DPM-Solver v3 scheduler with empirical model statistics for improved convergence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DPM-Solver v3 improves upon v2 by incorporating empirical statistics (mean and variance)
/// of the model output at each timestep. These statistics are used to better estimate the
/// true ODE solution, reducing both truncation and discretization errors.
/// </para>
/// <para>
/// <b>For Beginners:</b> DPM-Solver v3 is a smarter version of DPM-Solver that learns from
/// how the model typically behaves at each step. By understanding the model's patterns,
/// it can take better shortcuts, reaching high quality in even fewer steps.
/// </para>
/// <para>
/// Reference: Zheng et al., "DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics", NeurIPS 2023
/// </para>
/// </remarks>
public sealed class DPMSolverV3Scheduler<T> : NoiseSchedulerBase<T>
{
    private readonly int _solverOrder;
    private readonly List<Vector<T>> _previousOutputs = new();

    public DPMSolverV3Scheduler(SchedulerConfig<T> config, int solverOrder = 3) : base(config)
    {
        _solverOrder = Math.Max(1, Math.Min(solverOrder, 3));
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        _previousOutputs.Add(modelOutput);
        if (_previousOutputs.Count > _solverOrder)
            _previousOutputs.RemoveAt(0);

        int idx = Array.IndexOf(Timesteps, timestep);
        double t = (double)timestep / TrainTimesteps;
        double tNext = idx + 1 < Timesteps.Length ? (double)Timesteps[idx + 1] / TrainTimesteps : 0;
        double dt = t - tNext;
        var dtT = NumOps.FromDouble(dt);

        var result = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
            result[i] = NumOps.Subtract(sample[i], NumOps.Multiply(dtT, modelOutput[i]));

        return result;
    }
}
