namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Sets the learning rate using a cosine annealing schedule.
/// </summary>
/// <remarks>
/// <para>
/// CosineAnnealingLR uses a cosine function to smoothly decrease the learning rate from the
/// initial value to a minimum value over a specified number of steps. This is widely used
/// in modern deep learning and often outperforms step-based decay schedules.
/// </para>
/// <para><b>For Beginners:</b> Instead of making sudden drops in learning rate, cosine annealing
/// provides a smooth, curved decrease that follows the shape of a cosine wave. The learning rate
/// starts high, decreases slowly at first, then more rapidly in the middle, and finally slows
/// down again as it approaches the minimum. This smooth transition often leads to better model
/// performance than abrupt changes.
/// </para>
/// <para>
/// Formula: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * step / T_max))
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Cosine annealing over 100 epochs
/// var scheduler = new CosineAnnealingLRScheduler(
///     baseLearningRate: 0.1,
///     tMax: 100,
///     etaMin: 0.001
/// );
/// </code>
/// </example>
public class CosineAnnealingLRScheduler : LearningRateSchedulerBase
{
    private readonly int _tMax;
    private readonly double _etaMin;

    /// <summary>
    /// Initializes a new instance of the CosineAnnealingLRScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The initial (maximum) learning rate.</param>
    /// <param name="tMax">Maximum number of steps (typically total epochs or iterations).</param>
    /// <param name="etaMin">Minimum learning rate. Default: 0</param>
    /// <exception cref="ArgumentException">Thrown when tMax is not positive.</exception>
    public CosineAnnealingLRScheduler(
        double baseLearningRate,
        int tMax,
        double etaMin = 0.0)
        : base(baseLearningRate, etaMin)
    {
        if (tMax <= 0)
            throw new ArgumentException("T_max must be positive.", nameof(tMax));

        _tMax = tMax;
        _etaMin = etaMin;
    }

    /// <summary>
    /// Gets the maximum number of steps.
    /// </summary>
    public int TMax => _tMax;

    /// <summary>
    /// Gets the minimum learning rate.
    /// </summary>
    public double EtaMin => _etaMin;

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        // Clamp step to T_max for behavior after completion
        int effectiveStep = Math.Min(step, _tMax);

        double cosineValue = Math.Cos(Math.PI * effectiveStep / _tMax);
        return _etaMin + 0.5 * (_baseLearningRate - _etaMin) * (1 + cosineValue);
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["t_max"] = _tMax;
        state["eta_min"] = _etaMin;
        return state;
    }
}
