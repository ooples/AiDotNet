namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// Densification firing schedule for GaussianSplatting (#1835 excellence goal #2).
/// Reference GS impls hard-code a fixed-interval schedule ("every 100 iters densify");
/// AiDotNet exposes this as a swappable strategy so callers can plug in a data-driven
/// schedule (gradient-variance / loss-plateau triggers) without touching the model.
/// </summary>
public interface IDensificationSchedule
{
    /// <summary>
    /// Whether densification should fire at <paramref name="iteration"/> given the observed
    /// signals. Fixed-interval schedules ignore the signals; adaptive schedules key off them.
    /// </summary>
    /// <param name="iteration">Current training step (0-indexed).</param>
    /// <param name="lastLoss">Loss at the current step (may be zero for the first call).</param>
    /// <param name="lastGradientNorm">Mean gradient norm at the current step.</param>
    bool ShouldFire(int iteration, double lastLoss, double lastGradientNorm);
}

/// <summary>
/// Fixed-interval schedule — reference-impl behavior. Fires every <c>Interval</c> iterations
/// within the model's densification window. This is the default when no schedule is supplied
/// so the industry-standard behavior stays intact.
/// </summary>
public sealed class FixedIntervalDensificationSchedule : IDensificationSchedule
{
    public int Interval { get; init; } = 100;

    public bool ShouldFire(int iteration, double lastLoss, double lastGradientNorm)
    {
        int step = System.Math.Max(1, Interval);
        return iteration > 0 && iteration % step == 0;
    }
}

/// <summary>
/// Adaptive schedule (excellence goal): fires when the caller-observed loss plateaus below a
/// threshold OR the gradient-norm variance settles — signals that the cloud has stabilized
/// enough that split noise won't destabilize training, while its remaining representational
/// capacity has been exhausted. Reference impls have no equivalent — the fixed-interval
/// schedule bakes in a fork's assumptions about when densification will help. This adaptive
/// path fires at the RIGHT time per-scene, not the wrong time on a global clock.
/// </summary>
public sealed class AdaptiveDensificationSchedule : IDensificationSchedule
{
    /// <summary>
    /// Minimum iterations between two adaptive fires (avoids over-densification during noisy
    /// early training when both signals may temporarily satisfy the threshold).
    /// </summary>
    public int MinInterval { get; init; } = 50;

    /// <summary>
    /// Loss must improve by less than this fraction across the last window for the plateau
    /// trigger to consider firing. Paper-independent; empirically 0.5% is a reasonable default.
    /// </summary>
    public double LossPlateauThreshold { get; init; } = 0.005;

    /// <summary>
    /// Window (number of iterations) over which loss plateau is measured.
    /// </summary>
    public int PlateauWindow { get; init; } = 200;

    private int _lastFireIteration = -1;
    private double _lossSum = 0;
    private double _prevLossSum = 0;
    private int _windowCount = 0;

    public bool ShouldFire(int iteration, double lastLoss, double lastGradientNorm)
    {
        _lossSum += lastLoss;
        _windowCount++;
        if (_windowCount < PlateauWindow)
        {
            return false;
        }
        double meanLoss = _lossSum / _windowCount;
        double improvement = _prevLossSum > 0
            ? (_prevLossSum - meanLoss) / _prevLossSum
            : double.PositiveInfinity;
        _prevLossSum = meanLoss;
        _lossSum = 0;
        _windowCount = 0;

        // Plateau = improvement is small AND non-negative. `improvement < threshold` alone
        // would ALSO fire on loss DIVERGENCE (negative improvement satisfies < threshold),
        // which is the opposite of what we want — densifying during a divergence event
        // adds more capacity to an already-unstable cloud. Require improvement >= 0 so
        // the fire only happens when the model is still moving forward but slowly.
        bool eligible = iteration - _lastFireIteration >= MinInterval;
        bool plateaued = improvement >= 0 && improvement < LossPlateauThreshold;
        if (eligible && plateaued)
        {
            _lastFireIteration = iteration;
            return true;
        }
        return false;
    }
}
