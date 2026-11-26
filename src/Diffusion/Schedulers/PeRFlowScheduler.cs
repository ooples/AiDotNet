using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// PeRFlow (Piecewise Rectified Flow) scheduler for accelerated multi-segment flow sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PeRFlow divides the ODE trajectory into K segments, each with independently rectified
/// straight paths. By ensuring each segment is straight, the overall trajectory requires
/// fewer total steps while maintaining generation quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> PeRFlow breaks the generation process into segments, making each
/// segment's path as straight as possible. This is like straightening a curved road into
/// connected straight segments — each segment is easy to traverse quickly.
/// </para>
/// <para>
/// Reference: Yan et al., "PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator", 2024
/// </para>
/// </remarks>
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public sealed class PeRFlowScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly int _numSegments;

    /// <summary>
    /// Creates a PeRFlow scheduler with <paramref name="numSegments"/> piecewise-straight time windows.
    /// </summary>
    /// <param name="config">The scheduler configuration.</param>
    /// <param name="numSegments">The number of trained segments K (at least 1).</param>
    /// <remarks>
    /// Every segment boundary is a scheduled time. As in the reference scheduler
    /// (magic-research/piecewise-rectified-flow, <c>PeRFlowScheduler.set_timesteps</c>), a step count below K is
    /// raised to K, one step per window, so no step ever crosses a segment boundary.
    /// </remarks>
    public PeRFlowScheduler(SchedulerConfig<T> config, int numSegments = 4) : base(config)
    {
        _numSegments = Math.Max(1, numSegments);
    }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        // PeRFlow splits [0, 1] into equal time windows whose flows are straight, so every window boundary is a
        // scheduled time and the steps are spread evenly over the windows (earlier windows take the remainder).
        // As in the reference set_timesteps, a step count below the window count is raised to it (one step per
        // window); the window count itself cannot exceed the training timesteps.
        int windows = Math.Min(_numSegments, TrainTimesteps);
        if (inferenceSteps > 0 && inferenceSteps < windows)
            inferenceSteps = windows;
        base.SetTimesteps(inferenceSteps);
        var timesteps = new int[inferenceSteps];
        int index = 0;
        for (int w = windows - 1; w >= 0; w--)
        {
            int windowRank = windows - 1 - w;
            int stepsInWindow = inferenceSteps / windows + (windowRank < inferenceSteps % windows ? 1 : 0);
            double top = (double)(w + 1) / windows;
            double bottom = (double)w / windows;
            for (int s = 0; s < stepsInWindow; s++)
            {
                double t = top - (top - bottom) * s / stepsInWindow;
                int timestep = (int)Math.Round(t * (TrainTimesteps - 1));
                if (index > 0)
                    timestep = Math.Min(timestep, timesteps[index - 1] - 1);
                // Keep enough integer timesteps free below this entry for the remaining ones so the tail stays
                // strictly decreasing (the base class guarantees inferenceSteps <= TrainTimesteps).
                timesteps[index] = Math.Max(inferenceSteps - 1 - index, timestep);
                index++;
            }
        }
        SetTimestepArray(timesteps);
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        // Within a window the flow is straight, so the Euler step over the gap to the next scheduled time is exact.
        double dt = SchedulerTensorOps.FlowStepSize(Timesteps, timestep, TrainTimesteps, 1.0 / Math.Max(1, Timesteps.Length));
        return Engine.Subtract(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(dt)));
    }
}
