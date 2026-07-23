namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Adjusts the learning rate in response to whether the monitored metric improved: shrink while
/// improving, grow while stalled.
/// </summary>
/// <remarks>
/// <para>
/// This is the rule optimizers previously applied inline when <c>UseAdaptiveLearningRate</c> was
/// set — improving fitness multiplies the rate by <c>LearningRateDecay</c> (converging, so take
/// finer steps); stalling divides by it (stuck, so take larger ones); the result is clamped to
/// [min, max]. Expressing it as a scheduler means there is exactly ONE thing that writes the
/// learning rate. Previously the inline rule and any attached scheduler both wrote it, so the
/// adaptive branch silently overwrote the schedule on every step.
/// </para>
/// <para>
/// Note the direction is the opposite of reduce-on-plateau, and deliberately so:
/// <see cref="ReduceOnPlateauScheduler"/> reduces the rate when progress stalls (assuming the step
/// is too large to settle), whereas this raises it (assuming the step is too small to escape). Both
/// are legitimate; they encode different beliefs about why progress stopped.
/// </para>
/// <para>
/// <b>For Beginners:</b> The learning rate controls how big a step the model takes each update.
/// This adjusts it automatically: smaller steps while the model is improving so it can settle
/// precisely, larger steps when it is stuck so it can escape.
/// </para>
/// </remarks>
public class AdaptiveFitnessScheduler : LearningRateSchedulerBase
{
    private readonly double _decay;
    private readonly double _maxLearningRate;
    private readonly bool _higherIsBetter;
    private double _bestMetric;

    /// <summary>
    /// Creates an adaptive scheduler.
    /// </summary>
    /// <param name="baseLearningRate">The starting learning rate. Must be positive.</param>
    /// <param name="decay">
    /// The factor applied on improvement and inverted on stagnation. Must be in (0, 1) — a value at
    /// or above 1 would grow the rate while improving, inverting the rule.
    /// </param>
    /// <param name="minLearningRate">Lower clamp. Must not be negative.</param>
    /// <param name="maxLearningRate">Upper clamp. Must exceed <paramref name="minLearningRate"/>.</param>
    /// <param name="higherIsBetter">
    /// The direction of the metric being fed in. Defaults to false (a loss, where lower is better).
    /// Set true for scores like R² or accuracy — otherwise every improvement reads as a regression
    /// and the rule runs backwards, growing the rate exactly when it should shrink. Optimizers pass
    /// their fitness calculator's <c>IsHigherScoreBetter</c> here.
    /// </param>
    public AdaptiveFitnessScheduler(
        double baseLearningRate,
        double decay = 0.95,
        double minLearningRate = 1e-6,
        double maxLearningRate = 1.0,
        bool higherIsBetter = false)
        : base(baseLearningRate, minLearningRate)
    {
        if (!IsFiniteValue(baseLearningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(baseLearningRate), baseLearningRate,
                "Base learning rate must be a finite number.");
        }

        if (!IsFiniteValue(decay) || decay <= 0 || decay >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(decay), decay,
                "Decay must be a finite value in (0, 1): it shrinks the rate on improvement and is inverted " +
                "to grow it on stagnation, so a value >= 1 would reverse the rule.");
        }

        if (!IsFiniteValue(minLearningRate) || minLearningRate < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minLearningRate), minLearningRate,
                "Minimum learning rate must be a finite, non-negative number.");
        }

        if (!IsFiniteValue(maxLearningRate) || maxLearningRate <= minLearningRate)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLearningRate), maxLearningRate,
                $"Maximum learning rate must be a finite value exceeding the minimum ({minLearningRate}).");
        }

        if (baseLearningRate < minLearningRate || baseLearningRate > maxLearningRate)
        {
            throw new ArgumentOutOfRangeException(nameof(baseLearningRate), baseLearningRate,
                $"Base learning rate must lie within the clamp range [{minLearningRate}, {maxLearningRate}].");
        }

        _decay = decay;
        _maxLearningRate = maxLearningRate;
        _higherIsBetter = higherIsBetter;
        _bestMetric = higherIsBetter ? double.NegativeInfinity : double.PositiveInfinity;
    }

    /// <inheritdoc />
    public override double Step(double metric)
    {
        _currentStep++;

        // A non-finite metric (NaN or +/-Infinity) is a diverged/invalid observation. Treat it as
        // non-improving WITHOUT recording it as the best value: +Infinity would otherwise register as an
        // improvement under one metric direction and permanently poison the comparison history.
        if (!IsFiniteValue(metric))
        {
            _currentLearningRate /= _decay;
            _currentLearningRate = Math.Max(_minLearningRate, Math.Min(_maxLearningRate, _currentLearningRate));
            return _currentLearningRate;
        }

        bool improved = _higherIsBetter ? metric > _bestMetric : metric < _bestMetric;
        if (improved)
        {
            _bestMetric = metric;
            _currentLearningRate *= _decay;
        }
        else
        {
            _currentLearningRate /= _decay;
        }

        _currentLearningRate = Math.Max(_minLearningRate, Math.Min(_maxLearningRate, _currentLearningRate));
        return _currentLearningRate;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Without a metric there is nothing to adapt to, so the rate is left unchanged rather than
    /// drifting in an arbitrary direction. Drive this schedule with <see cref="Step(double)"/>.
    /// </remarks>
    public override double Step()
    {
        _currentStep++;
        return _currentLearningRate;
    }

    /// <inheritdoc />
    /// <remarks>
    /// The rate here depends on the history of metrics, not on the step index, so a rate for an
    /// arbitrary step cannot be computed without replaying that history.
    /// </remarks>
    protected override double ComputeLearningRate(int step) => _currentLearningRate;

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _bestMetric = _higherIsBetter ? double.NegativeInfinity : double.PositiveInfinity;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Extends the base rate/step state with the tracked best metric, so a scheduler restored from a
    /// checkpoint does not forget its improvement history and treat the next observation as the first.
    /// </remarks>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["best_metric"] = _bestMetric;
        return state;
    }

    /// <inheritdoc />
    public override void LoadState(Dictionary<string, object> state)
    {
        base.LoadState(state);
        if (state.TryGetValue("best_metric", out var bestMetric))
        {
            _bestMetric = Convert.ToDouble(bestMetric);
        }
    }

    private static bool IsFiniteValue(double v) => !double.IsNaN(v) && !double.IsInfinity(v);
}
