using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Stochastic Interpolant scheduler for generalized flow-based sampling with noise injection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements the Stochastic Interpolant framework which unifies flow matching and
/// score-based diffusion. Uses time-dependent interpolation between data and noise with
/// optional stochasticity controlled by an auxiliary noise schedule.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a flexible sampler that can behave like either a flow
/// model (deterministic, straight paths) or a diffusion model (stochastic, with randomness),
/// or anything in between. You can tune the "stochasticity dial" to find the sweet spot.
/// </para>
/// <para>
/// Reference: Albergo et al., "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions", 2023
/// </para>
/// </remarks>
public sealed class StochasticInterpolantScheduler<T> : NoiseSchedulerBase<T>
{
    private readonly double _stochasticity;

    public StochasticInterpolantScheduler(SchedulerConfig<T> config, double stochasticity = 0.5) : base(config)
    {
        _stochasticity = Math.Max(0.0, Math.Min(stochasticity, 1.0));
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

            if (noise != null && _stochasticity > 0)
            {
                double noiseLevel = _stochasticity * Math.Sqrt(dt * t);
                stepped = NumOps.Add(stepped, NumOps.Multiply(NumOps.FromDouble(noiseLevel), noise[i]));
            }

            result[i] = stepped;
        }

        return result;
    }
}
