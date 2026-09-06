namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Implements the tri-stage schedule: linear warmup, then a constant hold, then linear decay.
/// </summary>
/// <remarks>
/// <para>
/// For total steps <c>N</c>, warmup fraction <c>w</c>, hold fraction <c>h</c>, peak rate <c>p</c>
/// and floor <c>m</c>, with <c>W = wN</c> and <c>H = hN</c>:
/// <code>
/// m + (p - m) * t / W                     when t &lt; W          (linear warmup)
/// p                                       when W &lt;= t &lt; W + H  (hold)
/// p - (p - m) * (t - W - H) / (N - W - H) otherwise            (linear decay)
/// </code>
/// </para>
/// <para>
/// This is the schedule wav2vec 2.0 fine-tunes with — "a tri-state rate schedule where the
/// learning rate is warmed up for the first 10% of updates, held constant for the next 40% and
/// then linearly decayed for the remainder" (Baevski et al. 2020, Sec. 4.3) — and it is shared by
/// much of the speech literature that followed it.
/// </para>
/// <para>
/// It is deliberately its own scheduler rather than an approximation by an existing one. The
/// nearest available shapes are warmup-then-decay, which has no hold phase at all, and the
/// Noam-hold schedule, whose third stage decays as a power of the step rather than linearly. Both
/// produce a visibly different curve, and substituting a different curve for a published one is
/// harder to notice than having no schedule at all.
/// </para>
/// <para><b>For Beginners:</b> Three phases. The rate climbs from near zero so early, noisy
/// gradients cannot wreck the weights; it then sits at its peak for the bulk of training, where
/// most of the learning happens; and it finally falls back towards zero so the model settles
/// instead of bouncing around a minimum.
/// </para>
/// </remarks>
public class TriStageScheduler : LearningRateSchedulerBase
{
    private readonly int _warmupSteps;
    private readonly int _holdSteps;
    private readonly int _totalSteps;

    /// <summary>Creates a tri-stage schedule from explicit step counts.</summary>
    /// <param name="baseLearningRate">The peak rate, reached at the end of warmup.</param>
    /// <param name="warmupSteps">Steps spent ramping up to the peak.</param>
    /// <param name="holdSteps">Steps spent at the peak.</param>
    /// <param name="totalSteps">The whole run; decay fills whatever remains.</param>
    /// <param name="minLearningRate">The floor the decay ends at, and the rate warmup starts from.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// When any count is negative, or warmup and hold together exceed the run — a schedule whose
    /// decay phase has negative length is not a schedule, and silently clamping it would hide a
    /// mis-transcribed paper value rather than surface it.
    /// </exception>
    public TriStageScheduler(
        double baseLearningRate,
        int warmupSteps,
        int holdSteps,
        int totalSteps,
        double minLearningRate = 0.0)
        : base(baseLearningRate, minLearningRate)
    {
        if (warmupSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(warmupSteps), warmupSteps, "Warmup steps cannot be negative.");
        if (holdSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(holdSteps), holdSteps, "Hold steps cannot be negative.");
        if (totalSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalSteps), totalSteps, "Total steps must be positive.");
        if (warmupSteps + holdSteps > totalSteps)
        {
            throw new ArgumentOutOfRangeException(
                nameof(holdSteps), warmupSteps + holdSteps,
                $"Warmup ({warmupSteps}) and hold ({holdSteps}) together exceed the run ({totalSteps}), "
                + "leaving no room to decay.");
        }

        _warmupSteps = warmupSteps;
        _holdSteps = holdSteps;
        _totalSteps = totalSteps;
    }

    /// <summary>Creates a tri-stage schedule from the fractions a paper usually states.</summary>
    /// <remarks>
    /// Papers give these as proportions of the run ("the first 10% of updates"), which stays
    /// correct at any run length where a transcribed step count is correct at exactly one.
    /// </remarks>
    public static TriStageScheduler FromFractions(
        double baseLearningRate,
        int totalSteps,
        double warmupFraction,
        double holdFraction,
        double minLearningRate = 0.0)
        => new(baseLearningRate,
               (int)Math.Round(totalSteps * warmupFraction),
               (int)Math.Round(totalSteps * holdFraction),
               totalSteps,
               minLearningRate);

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        if (step < _warmupSteps)
        {
            // Starts at the floor rather than at zero so a declared floor is honoured in every
            // phase; with the usual floor of zero this is the plain linear ramp the papers describe.
            return _minLearningRate
                + (_baseLearningRate - _minLearningRate) * (step + 1) / _warmupSteps;
        }

        int decayStart = _warmupSteps + _holdSteps;
        if (step < decayStart) return _baseLearningRate;

        int decaySteps = _totalSteps - decayStart;
        if (decaySteps <= 0 || step >= _totalSteps) return _minLearningRate;

        double progress = (double)(step - decayStart) / decaySteps;
        return _baseLearningRate - (_baseLearningRate - _minLearningRate) * progress;
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["warmup_steps"] = _warmupSteps;
        state["hold_steps"] = _holdSteps;
        state["total_steps"] = _totalSteps;
        return state;
    }
}
