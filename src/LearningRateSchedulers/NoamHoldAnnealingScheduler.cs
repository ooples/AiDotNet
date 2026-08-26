namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Implements the extended Noam schedule introduced for Squeezeformer.
/// </summary>
/// <remarks>
/// For one-based optimizer step <c>t</c>, peak learning rate <c>p</c>, warmup
/// length <c>T0</c>, hold length <c>Th</c>, and decay power <c>d</c>:
/// <code>
/// p * t / T0                         when t &lt; T0
/// p                                  when T0 &lt;= t &lt; T0 + Th
/// p * T0^d / (t - Th)^d              otherwise
/// </code>
/// The ordinary Noam decay is the special case <c>Th = 0</c>, <c>d = 0.5</c>.
/// Squeezeformer uses 20 epochs of warmup, 160 epochs of hold, and <c>d = 1</c>.
/// </remarks>
public sealed class NoamHoldAnnealingScheduler : LearningRateSchedulerBase
{
    /// <summary>Gets the peak learning rate.</summary>
    public double PeakLearningRate => _baseLearningRate;

    /// <summary>Gets the number of linear-warmup optimizer steps.</summary>
    public int WarmupSteps { get; }

    /// <summary>Gets the number of optimizer steps that hold the peak rate.</summary>
    public int HoldSteps { get; }

    /// <summary>Gets the polynomial decay power.</summary>
    public double DecayRate { get; }

    /// <summary>Initializes an extended Noam warmup-hold-decay schedule.</summary>
    public NoamHoldAnnealingScheduler(
        double peakLearningRate,
        int warmupSteps,
        int holdSteps,
        double decayRate)
        : base(ValidatePeakLearningRate(peakLearningRate))
    {
        if (warmupSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(warmupSteps), "Warmup steps must be positive.");
        if (holdSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(holdSteps), "Hold steps must be non-negative.");
        if (double.IsNaN(decayRate) || double.IsInfinity(decayRate) || decayRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(decayRate), "Decay rate must be finite and positive.");

        WarmupSteps = warmupSteps;
        HoldSteps = holdSteps;
        DecayRate = decayRate;
        _currentLearningRate = ComputeLearningRate(0);
    }

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _currentLearningRate = ComputeLearningRate(0);
    }

    /// <inheritdoc />
    protected override double ComputeLearningRate(int step)
    {
        double trainingStep = step + 1.0;
        if (trainingStep < WarmupSteps)
            return PeakLearningRate * trainingStep / WarmupSteps;

        if (trainingStep < WarmupSteps + HoldSteps)
            return PeakLearningRate;

        double decayStep = trainingStep - HoldSteps;
        return PeakLearningRate * Math.Pow(WarmupSteps / decayStep, DecayRate);
    }

    /// <inheritdoc />
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["warmup_steps"] = WarmupSteps;
        state["hold_steps"] = HoldSteps;
        state["decay_rate"] = DecayRate;
        return state;
    }

    private static double ValidatePeakLearningRate(double peakLearningRate)
    {
        if (double.IsNaN(peakLearningRate) || double.IsInfinity(peakLearningRate) || peakLearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(peakLearningRate), "Peak learning rate must be finite and positive.");
        return peakLearningRate;
    }
}
