using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
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
        // Linear interpolant x_t = (1 - t) x_0 + t * eps with drift b = v = eps - x_0. The deterministic part is the
        // Euler step x_{t - dt} = x_t - dt * v over the gap to the next scheduled time.
        double t = SchedulerTensorOps.FlowTime(timestep, TrainTimesteps);
        double dt = SchedulerTensorOps.FlowStepSize(Timesteps, timestep, TrainTimesteps, 1.0 / Math.Max(1, Timesteps.Length));
        var result = Engine.Subtract(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(dt)));
        if (noise is null || _stochasticity <= 0 || t <= 0.0)
            return result;

        // Albergo et al. (2023) backward SDE dX = (b - eps(t) s) dt + sqrt(2 eps(t)) dW keeps the interpolant's
        // marginals. The noise std lambda * sqrt(dt * t) means eps(t) = lambda^2 t / 2, and with the score
        // s = -eps_hat / t (eps_hat = x_t + (1 - t) v) the drift correction is -(lambda^2 dt / 2) * eps_hat.
        var epsilon = Engine.Add(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(1.0 - t)));
        result = Engine.Subtract(result, Engine.Multiply(epsilon, NumOps.FromDouble(_stochasticity * _stochasticity * dt / 2.0)));
        return Engine.Add(result, Engine.Multiply(noise, NumOps.FromDouble(_stochasticity * Math.Sqrt(dt * t))));
    }
}
