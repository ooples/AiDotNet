namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Decays the learning rate using a polynomial function.
/// </summary>
/// <remarks>
/// <para>
/// PolynomialLR decays the learning rate from the initial value to a minimum value using
/// a polynomial function. The decay curve can be controlled by the power parameter -
/// power=1 gives linear decay, power&gt;1 gives faster initial decay, power&lt;1 gives slower initial decay.
/// </para>
/// <para><b>For Beginners:</b> This scheduler provides flexible control over how fast the learning
/// rate decreases. With power=1, it's a straight line decrease. With power=2, it decreases slowly
/// at first then more rapidly. With power=0.5, it decreases rapidly at first then slows down.
/// This flexibility lets you customize the decay curve to your specific training needs.
/// </para>
/// <para>
/// Formula: lr = (base_lr - end_lr) * (1 - step/total_steps)^power + end_lr
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Polynomial decay with power=2 (quadratic)
/// var scheduler = new PolynomialLRScheduler(
///     baseLearningRate: 0.1,
///     totalSteps: 100,
///     power: 2.0,
///     endLearningRate: 0.001
/// );
/// </code>
/// </example>
public class PolynomialLRScheduler : LearningRateSchedulerBase
{
    private readonly int _totalSteps;
    private readonly double _power;
    private readonly double _endLr;

    /// <summary>
    /// Initializes a new instance of the PolynomialLRScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The initial learning rate.</param>
    /// <param name="totalSteps">Total number of steps over which to decay.</param>
    /// <param name="power">The power of the polynomial. Default: 1.0 (linear)</param>
    /// <param name="endLearningRate">The final learning rate. Default: 0</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public PolynomialLRScheduler(
        double baseLearningRate,
        int totalSteps,
        double power = 1.0,
        double endLearningRate = 0.0)
        : base(baseLearningRate, endLearningRate)
    {
        if (totalSteps <= 0)
            throw new ArgumentException("Total steps must be positive.", nameof(totalSteps));
        if (power <= 0)
            throw new ArgumentException("Power must be positive.", nameof(power));

        _totalSteps = totalSteps;
        _power = power;
        _endLr = endLearningRate;
    }

    /// <summary>
    /// Gets the total number of steps.
    /// </summary>
    public int TotalSteps => _totalSteps;

    /// <summary>
    /// Gets the polynomial power.
    /// </summary>
    public double Power => _power;

    /// <summary>
    /// Gets the end learning rate.
    /// </summary>
    public double EndLearningRate => _endLr;

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        if (step >= _totalSteps)
        {
            return _endLr;
        }

        double progress = 1.0 - (double)step / _totalSteps;
        return (_baseLearningRate - _endLr) * Math.Pow(progress, _power) + _endLr;
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["total_steps"] = _totalSteps;
        state["power"] = _power;
        state["end_lr"] = _endLr;
        return state;
    }
}
