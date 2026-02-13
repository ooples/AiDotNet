namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Sets the learning rate using a user-defined lambda function.
/// </summary>
/// <remarks>
/// <para>
/// LambdaLR provides maximum flexibility by allowing you to define any learning rate schedule
/// as a function of the current step. The lambda function takes the step number and returns
/// a multiplier that is applied to the base learning rate.
/// </para>
/// <para><b>For Beginners:</b> This scheduler lets you define your own custom learning rate schedule
/// using a function. The function receives the current step number and returns a value that gets
/// multiplied with the initial learning rate. For example, returning 0.5 would give half the initial
/// learning rate. This is useful when you want a schedule that doesn't fit any of the standard patterns.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Custom schedule: lr = base_lr * (0.95 ^ epoch)
/// var scheduler = new LambdaLRScheduler(
///     baseLearningRate: 0.1,
///     lrLambda: step => Math.Pow(0.95, step)
/// );
///
/// // Warmup for 10 steps, then constant
/// var warmupScheduler = new LambdaLRScheduler(
///     baseLearningRate: 0.001,
///     lrLambda: step => step &lt; 10 ? (step + 1) / 10.0 : 1.0
/// );
/// </code>
/// </example>
public class LambdaLRScheduler : LearningRateSchedulerBase
{
    private readonly Func<int, double> _lrLambda;

    /// <summary>
    /// Initializes a new instance of the LambdaLRScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The initial learning rate.</param>
    /// <param name="lrLambda">A function that takes the step number and returns a multiplier for the base learning rate.</param>
    /// <param name="minLearningRate">Minimum learning rate floor. Default: 0</param>
    public LambdaLRScheduler(
        double baseLearningRate,
        Func<int, double> lrLambda,
        double minLearningRate = 0.0)
        : base(baseLearningRate, minLearningRate)
    {
        Guard.NotNull(lrLambda);
        _lrLambda = lrLambda;
    }

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        double multiplier = _lrLambda(step);
        return _baseLearningRate * multiplier;
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        // Note: Lambda function cannot be serialized
        state["scheduler_type"] = "LambdaLR";
        return state;
    }
}
