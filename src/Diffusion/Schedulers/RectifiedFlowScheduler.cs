using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Rectified flow scheduler for straight-path ODE sampling with velocity prediction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements rectified flow sampling where the model predicts velocity v = x_1 - x_0 along
/// straight paths between noise and data. Uses uniform time discretization for optimal
/// transport between distributions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Rectified flow draws the straightest possible path from noise to
/// image. This makes each step more efficient — you need fewer steps because you're not
/// following a curved path. Used by modern models like FLUX and SD3.
/// </para>
/// <para>
/// Reference: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023
/// </para>
/// </remarks>
public sealed class RectifiedFlowScheduler<T> : NoiseSchedulerBase<T>
{
    public RectifiedFlowScheduler(SchedulerConfig<T> config) : base(config) { }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        // Uniform time discretization from 1 to 0
        var timesteps = new int[inferenceSteps];
        for (int i = 0; i < inferenceSteps; i++)
            timesteps[i] = (int)((1.0 - (double)i / inferenceSteps) * (TrainTimesteps - 1));
        SetTimestepArray(timesteps);
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        // Rectified flow: x_{t-dt} = x_t - dt * v(x_t, t)
        double t = (double)timestep / TrainTimesteps;
        double dt = 1.0 / Timesteps.Length;
        var dtT = NumOps.FromDouble(dt);

        var result = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
            result[i] = NumOps.Subtract(sample[i], NumOps.Multiply(dtT, modelOutput[i]));

        return result;
    }
}
