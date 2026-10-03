using LrSchedule = AiDotNet.Tensors.Engines.Compilation.LrSchedule;

namespace AiDotNet.Optimizers.Fused;

/// <summary>
/// A fused-kernel learning-rate schedule that reads the eager optimizer's current rate on every step
/// instead of evaluating a schedule formula against the plan's step counter.
/// </summary>
/// <remarks>
/// <para>
/// The built-in <see cref="LrSchedule"/> shapes advance once per optimizer step. That matches a scheduler
/// stepped per batch, but not one stepped per epoch (<c>SchedulerStepMode.StepPerEpoch</c>, the default):
/// there the eager optimizer holds one rate for the whole epoch and the scheduler moves only at
/// <c>OnEpochEnd</c>. Mapping such a scheduler to a per-step shape annealed it over tMax batches rather than
/// tMax epochs, so the compiled path decayed the rate while the eager path held it.
/// </para>
/// <para>
/// The fused path does not call <c>OnBatchEnd</c>, so between epochs the optimizer's rate stays fixed, and
/// <c>OnEpochEnd</c> still advances it. Reading that rate here keeps the compiled update on exactly the
/// value the eager <c>Step</c> would use, for any scheduler type.
/// </para>
/// <para>
/// The plan evaluates <see cref="GetLr"/> outside any captured graph, once per step, so a changed rate is
/// seen on the next step. A plan holding this schedule cannot be checkpointed by the Tensors plan writer,
/// which AiDotNet does not use.
/// </para>
/// </remarks>
internal sealed class OptimizerTrackingLrSchedule : LrSchedule
{
    private readonly Func<double> _currentLearningRate;

    /// <summary>
    /// Creates a schedule that returns <paramref name="currentLearningRate"/> on every step.
    /// </summary>
    /// <param name="currentLearningRate">Reads the eager optimizer's current learning rate.</param>
    public OptimizerTrackingLrSchedule(Func<double> currentLearningRate)
    {
        _currentLearningRate = currentLearningRate
            ?? throw new ArgumentNullException(nameof(currentLearningRate));
    }

    /// <inheritdoc/>
    public override double GetLr(int step) => _currentLearningRate();
}
