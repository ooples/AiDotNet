using AiDotNet.Enums;

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
public sealed class PeRFlowScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly int _numSegments;

    public PeRFlowScheduler(SchedulerConfig<T> config, int numSegments = 4) : base(config)
    {
        _numSegments = Math.Max(1, numSegments);
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        double t = (double)timestep / TrainTimesteps;

        // Determine which segment we're in and compute local dt
        double segmentSize = 1.0 / _numSegments;
        int segmentIdx = Math.Min((int)(t / segmentSize), _numSegments - 1);
        double dt = segmentSize / Math.Max(1, Timesteps.Length / _numSegments);
        var dtT = NumOps.FromDouble(dt);

        var result = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
            result[i] = NumOps.Subtract(sample[i], NumOps.Multiply(dtT, modelOutput[i]));

        return result;
    }
}
