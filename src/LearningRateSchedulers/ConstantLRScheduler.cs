namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Maintains a constant learning rate throughout training.
/// </summary>
/// <remarks>
/// <para>
/// ConstantLR simply returns the same learning rate for every step. While this is the simplest
/// scheduler, it can be useful as a component in composite schedulers or for fine-tuning
/// where you want to keep the learning rate fixed.
/// </para>
/// <para><b>For Beginners:</b> This is the simplest scheduler - it just keeps the learning rate
/// the same throughout training. While adaptive schedules often work better, sometimes you want
/// a fixed learning rate, especially for fine-tuning or when the learning rate has already been
/// carefully tuned for your specific problem.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var scheduler = new ConstantLRScheduler(baseLearningRate: 0.001);
/// </code>
/// </example>
public class ConstantLRScheduler : LearningRateSchedulerBase
{
    /// <summary>
    /// Initializes a new instance of the ConstantLRScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The constant learning rate to maintain.</param>
    public ConstantLRScheduler(double baseLearningRate)
        : base(baseLearningRate)
    {
    }

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        return _baseLearningRate;
    }
}
