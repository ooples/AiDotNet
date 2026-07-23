namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Decays the learning rate exponentially every step.
/// </summary>
/// <remarks>
/// <para>
/// ExponentialLR decays the learning rate by gamma every step. This provides a smooth,
/// continuous decay that can be useful for certain training scenarios.
/// </para>
/// <para><b>For Beginners:</b> This scheduler smoothly reduces the learning rate at every step
/// by multiplying it by a factor (gamma). Unlike StepLR which makes sudden drops, exponential
/// decay provides a gradual, continuous reduction. Think of it like gradually releasing pressure
/// from a gas pedal rather than making sudden brake taps.
/// </para>
/// <para>
/// Formula: lr = base_lr * gamma^step
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Decay by 0.95 every epoch
/// var scheduler = new ExponentialLRScheduler(
///     baseLearningRate: 0.1,
///     gamma: 0.95
/// );
/// </code>
/// </example>
public class ExponentialLRScheduler : LearningRateSchedulerBase
{
    private readonly double _gamma;

    /// <summary>
    /// Initializes a new instance of the ExponentialLRScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The initial learning rate.</param>
    /// <param name="gamma">Multiplicative factor of learning rate decay per step. Default: 0.95</param>
    /// <param name="minLearningRate">Minimum learning rate floor. Default: 0</param>
    /// <exception cref="ArgumentException">Thrown when gamma is not in (0, 1].</exception>
    public ExponentialLRScheduler(
        double baseLearningRate,
        double gamma = 0.95,
        double minLearningRate = 0.0)
        : base(baseLearningRate, minLearningRate)
    {
        if (gamma <= 0 || gamma > 1)
            throw new ArgumentException("Gamma must be in (0, 1].", nameof(gamma));

        _gamma = gamma;
    }

    /// <summary>
    /// Gets the multiplicative factor of learning rate decay.
    /// </summary>
    public double Gamma => _gamma;

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        return _baseLearningRate * Math.Pow(_gamma, step);
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["gamma"] = _gamma;
        return state;
    }
}
