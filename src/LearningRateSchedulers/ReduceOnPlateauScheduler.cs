namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Reduces learning rate when a metric has stopped improving.
/// </summary>
/// <remarks>
/// <para>
/// ReduceOnPlateau monitors a quantity (usually validation loss) and reduces the learning rate
/// when no improvement is seen for a 'patience' number of evaluations. This is a reactive
/// scheduler that adapts based on training progress rather than a fixed schedule.
/// </para>
/// <para><b>For Beginners:</b> Unlike other schedulers that follow a fixed schedule, this one
/// watches your model's performance and only reduces the learning rate when training gets "stuck"
/// (plateaus). If the model keeps improving, it keeps the learning rate the same. If improvement
/// stops for a while (patience epochs), it reduces the learning rate to allow finer adjustments.
/// Think of it like slowing down only when you notice you're not making progress.
/// </para>
/// <para>
/// This scheduler requires you to call the Step(metric) overload with the monitored value.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var scheduler = new ReduceOnPlateauScheduler(
///     baseLearningRate: 0.1,
///     factor: 0.1,
///     patience: 10,
///     mode: ReduceOnPlateauScheduler.Mode.Min
/// );
///
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     Train(model, scheduler.CurrentLearningRate);
///     double valLoss = Validate(model);
///     scheduler.Step(valLoss);  // Scheduler decides whether to reduce LR
/// }
/// </code>
/// </example>
public class ReduceOnPlateauScheduler : LearningRateSchedulerBase
{
    private readonly double _factor;
    private readonly int _patience;
    private readonly double _threshold;
    private readonly ThresholdMode _thresholdMode;
    private readonly int _cooldown;
    private readonly Mode _mode;

    private double _bestValue;
    private int _badEpochs;
    private int _cooldownCounter;
    private int _numBadEpochs;

    /// <summary>
    /// Optimization mode.
    /// </summary>
    public enum Mode
    {
        /// <summary>Reduce LR when metric stops decreasing (for losses)</summary>
        Min,
        /// <summary>Reduce LR when metric stops increasing (for accuracies)</summary>
        Max
    }

    /// <summary>
    /// Threshold comparison mode.
    /// </summary>
    public enum ThresholdMode
    {
        /// <summary>Dynamic threshold: best * (1 + threshold) for max, best * (1 - threshold) for min</summary>
        Relative,
        /// <summary>Static threshold: best + threshold for max, best - threshold for min</summary>
        Absolute
    }

    /// <summary>
    /// Initializes a new instance of the ReduceOnPlateauScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The initial learning rate.</param>
    /// <param name="factor">Factor by which the learning rate is reduced. Default: 0.1</param>
    /// <param name="patience">Number of epochs with no improvement after which LR is reduced. Default: 10</param>
    /// <param name="threshold">Threshold for measuring improvement. Default: 1e-4</param>
    /// <param name="thresholdMode">How to compare with threshold. Default: Relative</param>
    /// <param name="cooldown">Number of epochs to wait before resuming normal operation after LR reduction. Default: 0</param>
    /// <param name="mode">Optimization mode (min or max). Default: Min</param>
    /// <param name="minLearningRate">Minimum learning rate floor. Default: 0</param>
    public ReduceOnPlateauScheduler(
        double baseLearningRate,
        double factor = 0.1,
        int patience = 10,
        double threshold = 1e-4,
        ThresholdMode thresholdMode = ThresholdMode.Relative,
        int cooldown = 0,
        Mode mode = Mode.Min,
        double minLearningRate = 0.0)
        : base(baseLearningRate, minLearningRate)
    {
        if (factor >= 1.0 || factor <= 0)
            throw new ArgumentException("Factor must be in (0, 1).", nameof(factor));
        if (patience < 0)
            throw new ArgumentException("Patience must be non-negative.", nameof(patience));
        if (cooldown < 0)
            throw new ArgumentException("Cooldown must be non-negative.", nameof(cooldown));

        _factor = factor;
        _patience = patience;
        _threshold = threshold;
        _thresholdMode = thresholdMode;
        _cooldown = cooldown;
        _mode = mode;

        _bestValue = mode == Mode.Min ? double.MaxValue : double.MinValue;
        _badEpochs = 0;
        _cooldownCounter = 0;
        _numBadEpochs = 0;
    }

    /// <summary>
    /// Gets the reduction factor.
    /// </summary>
    public double Factor => _factor;

    /// <summary>
    /// Gets the patience value.
    /// </summary>
    public int Patience => _patience;

    /// <summary>
    /// Gets the current number of bad epochs.
    /// </summary>
    public int NumBadEpochs => _numBadEpochs;

    /// <summary>
    /// Gets the best metric value seen so far.
    /// </summary>
    public double BestValue => _bestValue;

    /// <summary>
    /// Steps the scheduler with a metric value.
    /// </summary>
    /// <param name="metric">The monitored metric value (e.g., validation loss).</param>
    /// <returns>The current learning rate.</returns>
    public double Step(double metric)
    {
        _currentStep++;

        if (_cooldownCounter > 0)
        {
            _cooldownCounter--;
            _numBadEpochs = 0;
            return _currentLearningRate;
        }

        bool isImprovement = IsBetter(metric);

        if (isImprovement)
        {
            _bestValue = metric;
            _numBadEpochs = 0;
        }
        else
        {
            _numBadEpochs++;
        }

        if (_numBadEpochs > _patience)
        {
            ReduceLearningRate();
            _cooldownCounter = _cooldown;
            _numBadEpochs = 0;
        }

        return _currentLearningRate;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Note: For ReduceOnPlateau, the standard Step() without a metric does not reduce LR.
    /// Use Step(double metric) instead for proper functionality.
    /// </remarks>
    public override double Step()
    {
        _currentStep++;
        return _currentLearningRate;
    }

    private bool IsBetter(double current)
    {
        if (_mode == Mode.Min)
        {
            if (_thresholdMode == ThresholdMode.Relative)
            {
                return current < _bestValue * (1 - _threshold);
            }
            else
            {
                return current < _bestValue - _threshold;
            }
        }
        else
        {
            if (_thresholdMode == ThresholdMode.Relative)
            {
                return current > _bestValue * (1 + _threshold);
            }
            else
            {
                return current > _bestValue + _threshold;
            }
        }
    }

    private void ReduceLearningRate()
    {
        double newLr = _currentLearningRate * _factor;
        _currentLearningRate = Math.Max(_minLearningRate, newLr);
    }

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        // ReduceOnPlateau doesn't compute LR based on step
        // It's reactive based on metric values
        return _currentLearningRate;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _bestValue = _mode == Mode.Min ? double.MaxValue : double.MinValue;
        _badEpochs = 0;
        _cooldownCounter = 0;
        _numBadEpochs = 0;
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["factor"] = _factor;
        state["patience"] = _patience;
        state["threshold"] = _threshold;
        state["threshold_mode"] = _thresholdMode.ToString();
        state["cooldown"] = _cooldown;
        state["mode"] = _mode.ToString();
        state["best_value"] = _bestValue;
        state["bad_epochs"] = _badEpochs;
        state["cooldown_counter"] = _cooldownCounter;
        state["num_bad_epochs"] = _numBadEpochs;
        return state;
    }

    /// <inheritdoc/>
    public override void LoadState(Dictionary<string, object> state)
    {
        base.LoadState(state);
        if (state.TryGetValue("best_value", out var best))
            _bestValue = Convert.ToDouble(best);
        if (state.TryGetValue("bad_epochs", out var bad))
            _badEpochs = Convert.ToInt32(bad);
        if (state.TryGetValue("cooldown_counter", out var cool))
            _cooldownCounter = Convert.ToInt32(cool);
        if (state.TryGetValue("num_bad_epochs", out var numBad))
            _numBadEpochs = Convert.ToInt32(numBad);
    }
}
