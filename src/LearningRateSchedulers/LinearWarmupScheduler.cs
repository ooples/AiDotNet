namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Implements linear learning rate warmup followed by constant or decay schedule.
/// </summary>
/// <remarks>
/// <para>
/// Linear warmup gradually increases the learning rate from a small initial value to the
/// target learning rate over a specified number of warmup steps. This is commonly used
/// in transformer training and helps stabilize early training dynamics.
/// </para>
/// <para><b>For Beginners:</b> When training starts, the model's weights are random and
/// can produce large, unstable gradients. Starting with a very small learning rate and
/// gradually increasing it (warmup) helps the model stabilize before moving to the full
/// learning rate. Think of it like warming up an engine before driving at full speed.
/// </para>
/// <para>
/// This scheduler supports three modes after warmup:
/// - Constant: Keep the base learning rate after warmup
/// - Linear decay: Linearly decrease to a minimum value
/// - Cosine decay: Use cosine annealing to decrease to a minimum value
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Warmup for 1000 steps, then linear decay over remaining 9000 steps
/// var scheduler = new LinearWarmupScheduler(
///     baseLearningRate: 0.001,
///     warmupSteps: 1000,
///     totalSteps: 10000,
///     decayMode: LinearWarmupScheduler.DecayMode.Linear
/// );
/// </code>
/// </example>
public class LinearWarmupScheduler : LearningRateSchedulerBase
{
    private readonly int _warmupSteps;
    private readonly int _totalSteps;
    private readonly double _warmupInitLr;
    private readonly DecayMode _decayMode;
    private readonly double _endLr;

    /// <summary>
    /// Decay mode after warmup phase.
    /// </summary>
    public enum DecayMode
    {
        /// <summary>Keep constant learning rate after warmup</summary>
        Constant,
        /// <summary>Linear decay to minimum after warmup</summary>
        Linear,
        /// <summary>Cosine decay to minimum after warmup</summary>
        Cosine
    }

    /// <summary>
    /// Initializes a new instance of the LinearWarmupScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The target learning rate after warmup.</param>
    /// <param name="warmupSteps">Number of warmup steps.</param>
    /// <param name="totalSteps">Total number of training steps (required for decay modes).</param>
    /// <param name="warmupInitLr">Initial learning rate at start of warmup. Default: 0</param>
    /// <param name="decayMode">Decay mode after warmup. When null, automatically selects Linear decay
    /// if endLr differs from baseLearningRate, otherwise uses Constant. Default: null (auto-detect)</param>
    /// <param name="endLr">Final learning rate after decay. Default: 0</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public LinearWarmupScheduler(
        double baseLearningRate,
        int warmupSteps,
        int totalSteps = 0,
        double warmupInitLr = 0.0,
        DecayMode? decayMode = null,
        double endLr = 0.0)
        : base(baseLearningRate, endLr)
    {
        if (warmupSteps < 0)
            throw new ArgumentException("Warmup steps cannot be negative.", nameof(warmupSteps));

        _warmupSteps = warmupSteps;
        _totalSteps = totalSteps > 0 ? totalSteps : warmupSteps;
        _warmupInitLr = warmupInitLr;
        _endLr = endLr;

        // Facade pattern: auto-detect decay mode when not specified
        // If endLr differs from baseLearningRate, user likely wants decay - default to Linear
        // If user explicitly sets decayMode, respect their choice
        bool needsDecay = Math.Abs(endLr - baseLearningRate) > 1e-10 && totalSteps > warmupSteps;
        _decayMode = decayMode ?? (needsDecay ? DecayMode.Linear : DecayMode.Constant);

        if (totalSteps < warmupSteps && _decayMode != DecayMode.Constant)
            throw new ArgumentException("Total steps must be >= warmup steps for decay modes.", nameof(totalSteps));

        // Start at warmup initial learning rate
        _currentLearningRate = warmupInitLr;
    }

    /// <summary>
    /// Gets the number of warmup steps.
    /// </summary>
    public int WarmupSteps => _warmupSteps;

    /// <summary>
    /// Gets the total number of steps.
    /// </summary>
    public int TotalSteps => _totalSteps;

    /// <summary>
    /// Gets the decay mode.
    /// </summary>
    public DecayMode CurrentDecayMode => _decayMode;

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        if (step < _warmupSteps)
        {
            // Warmup phase: linear increase
            if (_warmupSteps == 0) return _baseLearningRate;
            double progress = (double)step / _warmupSteps;
            return _warmupInitLr + (_baseLearningRate - _warmupInitLr) * progress;
        }

        if (_decayMode == DecayMode.Constant)
        {
            return _baseLearningRate;
        }

        // Decay phase
        int decaySteps = _totalSteps - _warmupSteps;
        int decayStep = step - _warmupSteps;

        if (decayStep >= decaySteps)
        {
            return _endLr;
        }

        double decayProgress = (double)decayStep / decaySteps;

        if (_decayMode == DecayMode.Linear)
        {
            return _baseLearningRate - (_baseLearningRate - _endLr) * decayProgress;
        }
        else // Cosine
        {
            double cosineValue = (1 + Math.Cos(Math.PI * decayProgress)) / 2;
            return _endLr + (_baseLearningRate - _endLr) * cosineValue;
        }
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["warmup_steps"] = _warmupSteps;
        state["total_steps"] = _totalSteps;
        state["warmup_init_lr"] = _warmupInitLr;
        state["decay_mode"] = _decayMode.ToString();
        state["end_lr"] = _endLr;
        return state;
    }
}
