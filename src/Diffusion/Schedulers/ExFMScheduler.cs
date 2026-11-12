using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// ExFM (Exponential Flow Matching) scheduler with exponential time discretization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Uses a non-uniform (quadratic, t_i = (1 - i/N)^2) time discretization instead of uniform spacing
/// for flow matching. This concentrates more steps near t=0 (clean data) where fine details matter
/// and fewer steps near t=1 (noise) where coarse structure is determined.
/// </para>
/// <para>
/// <b>For Beginners:</b> ExFM places more computation effort on the final "polishing" steps
/// where fine details are added, and less on the early steps where just the rough shape
/// forms. This produces sharper images with the same number of total steps.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public sealed class ExFMScheduler<T> : NoiseSchedulerBase<T>
{
    public ExFMScheduler(SchedulerConfig<T> config) : base(config) { }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        // Quadratic time discretization t_i = (1 - i/N)^2: the gaps (2(N - i) - 1) / N^2 shrink towards t = 0,
        // so more steps land near the clean end where fine details form. Timesteps stay strictly decreasing.
        var timesteps = new int[inferenceSteps];
        for (int i = 0; i < inferenceSteps; i++)
        {
            double u = (double)i / inferenceSteps;
            double t = (1.0 - u) * (1.0 - u);
            int timestep = (int)(t * (TrainTimesteps - 1));
            if (i > 0)
                timestep = Math.Min(timestep, timesteps[i - 1] - 1);
            timesteps[i] = Math.Max(0, timestep);
        }
        SetTimestepArray(timesteps);
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        // Euler step of the flow ODE x_{t - dt} = x_t - dt * v over the actual gap to the next scheduled time:
        // the schedule is non-uniform, so a fixed 1/N step would over- and under-shoot the scheduled times.
        double dt = SchedulerTensorOps.FlowStepSize(Timesteps, timestep, TrainTimesteps, 1.0 / Math.Max(1, Timesteps.Length));
        return Engine.Subtract(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(dt)));
    }
}
