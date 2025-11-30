namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Decays the learning rate by a factor (gamma) every specified number of steps.
/// </summary>
/// <remarks>
/// <para>
/// StepLR is one of the simplest and most commonly used learning rate schedulers.
/// It multiplies the learning rate by gamma every step_size epochs/steps.
/// </para>
/// <para><b>For Beginners:</b> This scheduler reduces the learning rate by a fixed amount
/// at regular intervals. For example, you might reduce the learning rate by 10x every 30 epochs.
/// This is like slowing down periodically as you get closer to your destination, making
/// your adjustments more precise as training progresses.
/// </para>
/// <para>
/// Formula: lr = base_lr * gamma^(floor(step / step_size))
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Reduce LR by 10x every 30 epochs
/// var scheduler = new StepLRScheduler(
///     baseLearningRate: 0.1,
///     stepSize: 30,
///     gamma: 0.1
/// );
///
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     Train(model, scheduler.CurrentLearningRate);
///     scheduler.Step();
/// }
/// </code>
/// </example>
public class StepLRScheduler : LearningRateSchedulerBase
{
    private readonly int _stepSize;
    private readonly double _gamma;

    /// <summary>
    /// Initializes a new instance of the StepLRScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The initial learning rate.</param>
    /// <param name="stepSize">Period of learning rate decay (number of steps between each decay).</param>
    /// <param name="gamma">Multiplicative factor of learning rate decay. Default: 0.1</param>
    /// <param name="minLearningRate">Minimum learning rate floor. Default: 0</param>
    /// <exception cref="ArgumentException">Thrown when stepSize is not positive or gamma is not in (0, 1].</exception>
    public StepLRScheduler(
        double baseLearningRate,
        int stepSize,
        double gamma = 0.1,
        double minLearningRate = 0.0)
        : base(baseLearningRate, minLearningRate)
    {
        if (stepSize <= 0)
            throw new ArgumentException("Step size must be positive.", nameof(stepSize));
        if (gamma <= 0 || gamma > 1)
            throw new ArgumentException("Gamma must be in (0, 1].", nameof(gamma));

        _stepSize = stepSize;
        _gamma = gamma;
    }

    /// <summary>
    /// Gets the step size (period of learning rate decay).
    /// </summary>
    public int StepSize => _stepSize;

    /// <summary>
    /// Gets the multiplicative factor of learning rate decay.
    /// </summary>
    public double Gamma => _gamma;

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        int decayCount = step / _stepSize;
        return _baseLearningRate * Math.Pow(_gamma, decayCount);
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["step_size"] = _stepSize;
        state["gamma"] = _gamma;
        return state;
    }
}
