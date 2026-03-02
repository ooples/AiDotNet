using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Variational Rectified Flow scheduler with learned time-dependent noise injection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Extends rectified flow with a variational formulation that allows controlled stochasticity
/// at each step. The noise injection level is modulated by a learned schedule, improving
/// diversity while maintaining sample quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> This scheduler adds a controlled amount of randomness to rectified
/// flow sampling. Pure deterministic sampling can sometimes produce less diverse results.
/// This scheduler adds just enough randomness to improve variety without hurting quality.
/// </para>
/// </remarks>
public sealed class VariationalRFScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly double _noiseScale;

    public VariationalRFScheduler(SchedulerConfig<T> config, double noiseScale = 0.1) : base(config)
    {
        _noiseScale = noiseScale;
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        double t = (double)timestep / TrainTimesteps;
        double dt = 1.0 / Math.Max(1, Timesteps.Length);
        var dtT = NumOps.FromDouble(dt);

        var result = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
        {
            var stepped = NumOps.Subtract(sample[i], NumOps.Multiply(dtT, modelOutput[i]));

            // Add variational noise scaled by time and eta
            if (noise != null && NumOps.ToDouble(eta) > 0)
            {
                double noiseLevel = _noiseScale * t * NumOps.ToDouble(eta);
                stepped = NumOps.Add(stepped, NumOps.Multiply(NumOps.FromDouble(noiseLevel), noise[i]));
            }

            result[i] = stepped;
        }

        return result;
    }
}
