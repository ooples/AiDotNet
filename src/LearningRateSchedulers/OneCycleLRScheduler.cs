namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Implements the 1cycle learning rate policy.
/// </summary>
/// <remarks>
/// <para>
/// The 1cycle policy starts with a low learning rate, increases it to a maximum, then
/// decreases it again. This approach has been shown to enable faster training and
/// better final performance, especially when combined with momentum cycling.
/// </para>
/// <para><b>For Beginners:</b> The 1cycle policy is like warming up before a workout,
/// going full intensity during the workout, and then cooling down. The learning rate
/// starts low (warmup), ramps up to a maximum (peak training), and then decreases
/// to very low values (fine-tuning). This approach often allows training with higher
/// maximum learning rates and can achieve better results in fewer epochs.
/// </para>
/// <para>
/// Based on the paper "Super-Convergence: Very Fast Training of Neural Networks Using
/// Large Learning Rates" by Leslie N. Smith and Nicholay Topin.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // 1cycle policy over 100 epochs with peak LR of 0.1
/// var scheduler = new OneCycleLRScheduler(
///     maxLearningRate: 0.1,
///     totalSteps: 100,
///     pctStart: 0.3,      // 30% warmup
///     divFactor: 25,      // Start LR = 0.1/25 = 0.004
///     finalDivFactor: 1e4 // End LR = 0.1/10000 = 0.00001
/// );
/// </code>
/// </example>
public class OneCycleLRScheduler : LearningRateSchedulerBase
{
    private readonly double _maxLearningRate;
    private readonly int _totalSteps;
    private readonly double _pctStart;
    private readonly double _divFactor;
    private readonly double _finalDivFactor;
    private readonly AnnealingStrategy _annealStrategy;

    private readonly int _warmupSteps;
    private readonly int _annealSteps;
    private readonly double _initialLr;
    private readonly double _finalLr;

    /// <summary>
    /// Annealing strategy for the decay phase.
    /// </summary>
    public enum AnnealingStrategy
    {
        /// <summary>Cosine annealing (smooth decay)</summary>
        Cosine,
        /// <summary>Linear annealing (constant decay rate)</summary>
        Linear
    }

    /// <summary>
    /// Initializes a new instance of the OneCycleLRScheduler class.
    /// </summary>
    /// <param name="maxLearningRate">The maximum learning rate (peak of the cycle).</param>
    /// <param name="totalSteps">Total number of steps (typically epochs * steps_per_epoch).</param>
    /// <param name="pctStart">Percentage of the cycle spent increasing the learning rate. Default: 0.3</param>
    /// <param name="divFactor">Factor to determine initial learning rate (initial_lr = max_lr / div_factor). Default: 25</param>
    /// <param name="finalDivFactor">Factor to determine final learning rate (final_lr = initial_lr / final_div_factor). Default: 10000</param>
    /// <param name="annealStrategy">Annealing strategy for the decay phase. Default: Cosine</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public OneCycleLRScheduler(
        double maxLearningRate,
        int totalSteps,
        double pctStart = 0.3,
        double divFactor = 25.0,
        double finalDivFactor = 10000.0,
        AnnealingStrategy annealStrategy = AnnealingStrategy.Cosine)
        : base(maxLearningRate / divFactor)
    {
        if (maxLearningRate <= 0)
            throw new ArgumentException("Max learning rate must be positive.", nameof(maxLearningRate));
        if (totalSteps <= 0)
            throw new ArgumentException("Total steps must be positive.", nameof(totalSteps));
        if (pctStart < 0 || pctStart >= 1)
            throw new ArgumentException("pct_start must be in [0, 1).", nameof(pctStart));
        if (divFactor <= 0)
            throw new ArgumentException("div_factor must be positive.", nameof(divFactor));
        if (finalDivFactor <= 0)
            throw new ArgumentException("final_div_factor must be positive.", nameof(finalDivFactor));

        _maxLearningRate = maxLearningRate;
        _totalSteps = totalSteps;
        _pctStart = pctStart;
        _divFactor = divFactor;
        _finalDivFactor = finalDivFactor;
        _annealStrategy = annealStrategy;

        _warmupSteps = (int)(totalSteps * pctStart);
        _annealSteps = totalSteps - _warmupSteps;
        _initialLr = maxLearningRate / divFactor;
        _finalLr = _initialLr / finalDivFactor;
    }

    /// <summary>
    /// Gets the maximum learning rate.
    /// </summary>
    public double MaxLearningRate => _maxLearningRate;

    /// <summary>
    /// Gets the total number of steps.
    /// </summary>
    public int TotalSteps => _totalSteps;

    /// <summary>
    /// Gets the percentage of steps for warmup.
    /// </summary>
    public double PctStart => _pctStart;

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        if (step >= _totalSteps)
        {
            return _finalLr;
        }

        if (step < _warmupSteps)
        {
            // Warmup phase: linear increase from initial_lr to max_lr
            double progress = (double)step / _warmupSteps;
            return _initialLr + (_maxLearningRate - _initialLr) * progress;
        }
        else
        {
            // Annealing phase: decrease from max_lr to final_lr
            int annealStep = step - _warmupSteps;
            double progress = (double)annealStep / _annealSteps;

            if (_annealStrategy == AnnealingStrategy.Cosine)
            {
                // Cosine annealing
                double cosineValue = (1 + Math.Cos(Math.PI * progress)) / 2;
                return _finalLr + (_maxLearningRate - _finalLr) * cosineValue;
            }
            else
            {
                // Linear annealing
                return _maxLearningRate - (_maxLearningRate - _finalLr) * progress;
            }
        }
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["max_learning_rate"] = _maxLearningRate;
        state["total_steps"] = _totalSteps;
        state["pct_start"] = _pctStart;
        state["div_factor"] = _divFactor;
        state["final_div_factor"] = _finalDivFactor;
        state["anneal_strategy"] = _annealStrategy.ToString();
        return state;
    }
}
