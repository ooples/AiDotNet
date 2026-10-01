namespace AiDotNet.Optimizers.Fused;

/// <summary>
/// A fused-plan learning-rate schedule that returns the optimizer's current learning rate, which the optimizer's own
/// scheduler moves at its own cadence (epoch ends, for <c>SchedulerStepMode.StepPerEpoch</c>).
/// </summary>
/// <remarks>
/// The compiled plan evaluates its schedule once per optimizer step. A per-epoch schedule mapped to the plan's built-in
/// shapes would advance every batch, while the eager step holds the rate for the whole epoch. Changing the rate by
/// reconfiguring the plan is not an option either, because that resets its moments. Reading the host's rate on each
/// step holds it exactly as the eager step does, and it passes the rate in double precision rather than as a float.
/// </remarks>
internal sealed class HostHeldLrSchedule : Tensors.Engines.Compilation.LrSchedule
{
    private readonly Func<double> _currentRate;

    internal HostHeldLrSchedule(Func<double> currentRate)
    {
        _currentRate = currentRate ?? throw new ArgumentNullException(nameof(currentRate));
    }

    /// <inheritdoc />
    public override double GetLr(int step) => _currentRate();
}
