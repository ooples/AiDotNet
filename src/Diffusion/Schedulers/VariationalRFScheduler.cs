using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
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
        // Euler step x_{t - dt} = x_t - dt * v of the rectified flow over the gap to the next scheduled time.
        double t = SchedulerTensorOps.FlowTime(timestep, TrainTimesteps);
        double dt = SchedulerTensorOps.FlowStepSize(Timesteps, timestep, TrainTimesteps, 1.0 / Math.Max(1, Timesteps.Length));
        var result = Engine.Subtract(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(dt)));
        if (noise is null || !NumOps.GreaterThan(eta, NumOps.Zero) || t <= 0.0)
            return result;

        // Noise injection as a discretised SDE, so the amount of noise does not grow with the step count and the
        // marginals of the flow are kept: diffusion g(t) = noiseScale * eta * t gives the Brownian increment
        // g * sqrt(dt) * z and, with the score -eps_hat / t (eps_hat = x_t + (1 - t) v), the drift correction
        // -(g^2 dt / 2t) * eps_hat.
        T g = NumOps.Multiply(eta, NumOps.FromDouble(_noiseScale * t));
        var epsilon = Engine.Add(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(1.0 - t)));
        T scoreScale = NumOps.Multiply(NumOps.Multiply(g, g), NumOps.FromDouble(dt / (2.0 * t)));
        result = Engine.Subtract(result, Engine.Multiply(epsilon, scoreScale));
        return Engine.Add(result, Engine.Multiply(noise, NumOps.Multiply(g, NumOps.FromDouble(Math.Sqrt(dt)))));
    }
}
