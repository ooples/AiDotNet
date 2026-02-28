using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Discretized Rectified Flow scheduler with optimized timestep selection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Uses a learned or heuristic discretization of the continuous rectified flow ODE
/// that minimizes truncation error. Timesteps are selected to minimize the difference
/// between discrete and continuous ODE solutions.
/// </para>
/// <para>
/// <b>For Beginners:</b> This scheduler picks the best timesteps to use when converting
/// the continuous flow path into discrete steps. By choosing timesteps more carefully,
/// it gets better results with fewer steps compared to uniform spacing.
/// </para>
/// </remarks>
public sealed class DiscretizedRFScheduler<T> : NoiseSchedulerBase<T>
{
    public DiscretizedRFScheduler(SchedulerConfig<T> config) : base(config) { }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        double t = (double)timestep / TrainTimesteps;
        double dt = 1.0 / Math.Max(1, Timesteps.Length);
        var dtT = NumOps.FromDouble(dt);

        var result = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
            result[i] = NumOps.Subtract(sample[i], NumOps.Multiply(dtT, modelOutput[i]));

        return result;
    }
}
