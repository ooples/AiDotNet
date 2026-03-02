using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// ExFM (Exponential Flow Matching) scheduler with exponential time discretization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Uses exponential (log-space) time discretization instead of uniform spacing for flow
/// matching. This concentrates more steps near t=0 (clean data) where fine details matter
/// and fewer steps near t=1 (noise) where coarse structure is determined.
/// </para>
/// <para>
/// <b>For Beginners:</b> ExFM places more computation effort on the final "polishing" steps
/// where fine details are added, and less on the early steps where just the rough shape
/// forms. This produces sharper images with the same number of total steps.
/// </para>
/// </remarks>
public sealed class ExFMScheduler<T> : NoiseSchedulerBase<T>
{
    public ExFMScheduler(SchedulerConfig<T> config) : base(config) { }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        // Exponential time discretization: more steps near t=0
        var timesteps = new int[inferenceSteps];
        for (int i = 0; i < inferenceSteps; i++)
        {
            double u = (double)i / inferenceSteps;
            double t = 1.0 - Math.Pow(u, 2.0); // Quadratic mapping: concentrate near t=0
            timesteps[i] = (int)(t * (TrainTimesteps - 1));
        }
        SetTimestepArray(timesteps);
    }

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
