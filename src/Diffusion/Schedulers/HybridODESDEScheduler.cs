using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
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
        // Flow path x_t = (1 - t) x_0 + t * eps with velocity v = eps - x_0. The ODE phase is the Euler step
        // x_{t - dt} = x_t - dt * v over the gap to the next scheduled time.
        double t = SchedulerTensorOps.FlowTime(timestep, TrainTimesteps);
        double dt = SchedulerTensorOps.FlowStepSize(Timesteps, timestep, TrainTimesteps, 1.0 / Math.Max(1, Timesteps.Length));
        var result = Engine.Subtract(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(dt)));

        bool useSDE = t < _transitionPoint;
        if (!useSDE || noise is null || t <= 0.0)
            return result;

        // SDE phase: the reverse-time SDE dx = [v - (g^2 / 2) * score] dt + g dW with diffusion g = eta keeps the
        // marginals of the ODE (Ma et al., SiT 2024). On this path score = -eps / t with eps = x_t + (1 - t) v,
        // so the step adds -(g^2 dt / 2t) * eps plus the Brownian increment g * sqrt(dt) * z.
        var epsilon = Engine.Add(sample, Engine.Multiply(modelOutput, NumOps.FromDouble(1.0 - t)));
        T scoreScale = NumOps.Multiply(NumOps.Multiply(eta, eta), NumOps.FromDouble(dt / (2.0 * t)));
        result = Engine.Subtract(result, Engine.Multiply(epsilon, scoreScale));
        T noiseScale = NumOps.Multiply(eta, NumOps.FromDouble(Math.Sqrt(dt)));
        return Engine.Add(result, Engine.Multiply(noise, noiseScale));
    }
}
