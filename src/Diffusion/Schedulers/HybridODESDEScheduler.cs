using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Hybrid ODE/SDE scheduler that transitions between deterministic and stochastic sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Begins with deterministic ODE sampling for coarse structure, then switches to
/// stochastic SDE sampling for fine detail generation. The transition point is
/// configurable, balancing consistency with diversity.
/// </para>
/// <para>
/// <b>For Beginners:</b> This scheduler starts deterministically (same seed = same image)
/// for the main structure, then adds some controlled randomness for fine details.
/// This gives you both reliable composition and natural-looking textures.
/// </para>
/// </remarks>
public sealed class HybridODESDEScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly double _transitionPoint;

    public HybridODESDEScheduler(SchedulerConfig<T> config, double transitionPoint = 0.3) : base(config)
    {
        _transitionPoint = Math.Max(0.0, Math.Min(transitionPoint, 1.0));
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        double t = (double)timestep / TrainTimesteps;
        double dt = 1.0 / Math.Max(1, Timesteps.Length);
        var dtT = NumOps.FromDouble(dt);

        var result = new Vector<T>(sample.Length);
        bool useSDE = t < _transitionPoint;

        for (int i = 0; i < sample.Length; i++)
        {
            var stepped = NumOps.Subtract(sample[i], NumOps.Multiply(dtT, modelOutput[i]));

            // Add SDE noise in the detail phase
            if (useSDE && noise != null)
            {
                double noiseLevel = Math.Sqrt(dt) * NumOps.ToDouble(eta);
                stepped = NumOps.Add(stepped, NumOps.Multiply(NumOps.FromDouble(noiseLevel), noise[i]));
            }

            result[i] = stepped;
        }

        return result;
    }
}
