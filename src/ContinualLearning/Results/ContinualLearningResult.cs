namespace AiDotNet.ContinualLearning.Results;

/// <summary>
/// Result from training on a single task in continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class captures everything that happened during training
/// on one task - how well the model learned, how long it took, and whether it forgot
/// previous knowledge.</para>
///
/// <para><b>Key Metrics Explained:</b>
/// <list type="bullet">
/// <item><description><b>Training Loss:</b> How far predictions were from targets (lower is better)</description></item>
/// <item><description><b>Training Accuracy:</b> Percentage of correct predictions on training data</description></item>
/// <item><description><b>Average Previous Task Accuracy:</b> How well the model still performs on old tasks</description></item>
/// <item><description><b>Forgetting:</b> How much accuracy was lost on previous tasks (lower is better)</description></item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Parisi et al. "Continual Lifelong Learning with Neural Networks: A Review" (2019)</para>
/// </remarks>
public class ContinualLearningResult<T>
{
    /// <summary>
    /// Gets the task identifier (0-indexed).
    /// </summary>
    public int TaskId { get; }

    /// <summary>
    /// Gets the final training loss on this task.
    /// </summary>
    public T TrainingLoss { get; }

    /// <summary>
    /// Gets the final training accuracy on this task.
    /// </summary>
    public T TrainingAccuracy { get; }

    /// <summary>
    /// Gets the average accuracy on all previously learned tasks after training on this task.
    /// </summary>
    /// <remarks>
    /// <para>This is a key metric for detecting catastrophic forgetting.
    /// If this drops significantly after learning a new task, the model is forgetting.</para>
    /// </remarks>
    public T AveragePreviousTaskAccuracy { get; }

    /// <summary>
    /// Gets the time taken to train on this task.
    /// </summary>
    public TimeSpan TrainingTime { get; }

    /// <summary>
    /// Gets the loss history across training epochs.
    /// </summary>
    public Vector<T> LossHistory { get; }

    /// <summary>
    /// Gets the regularization loss history (e.g., EWC penalty term).
    /// </summary>
    public Vector<T> RegularizationLossHistory { get; }

    /// <summary>
    /// Gets the validation loss on this task, if validation was performed.
    /// </summary>
    public T? ValidationLoss { get; init; }

    /// <summary>
    /// Gets the validation accuracy on this task, if validation was performed.
    /// </summary>
    public T? ValidationAccuracy { get; init; }

    /// <summary>
    /// Gets the amount of forgetting on previous tasks (accuracy drop).
    /// </summary>
    /// <remarks>
    /// <para>Forgetting = (accuracy before learning this task) - (accuracy after learning this task)
    /// for each previous task, averaged. Positive values indicate forgetting.</para>
    /// </remarks>
    public T? Forgetting { get; init; }

    /// <summary>
    /// Gets the forward transfer metric.
    /// </summary>
    /// <remarks>
    /// <para>Forward transfer measures how much learning previous tasks helps with learning
    /// the current task. Positive values indicate positive transfer (prior learning helped).</para>
    /// </remarks>
    public T? ForwardTransfer { get; init; }

    /// <summary>
    /// Gets the number of samples used for training.
    /// </summary>
    public int? SampleCount { get; init; }

    /// <summary>
    /// Gets the peak memory usage during training in bytes.
    /// </summary>
    public long? PeakMemoryBytes { get; init; }

    /// <summary>
    /// Gets the number of gradient updates performed.
    /// </summary>
    public int? GradientUpdates { get; init; }

    /// <summary>
    /// Gets the effective learning rate used (may vary with schedulers).
    /// </summary>
    public T? EffectiveLearningRate { get; init; }

    /// <summary>
    /// Gets additional strategy-specific metrics.
    /// </summary>
    /// <remarks>
    /// <para>Different strategies track different metrics:
    /// - EWC: Fisher Information norm, regularization strength
    /// - GEM: Number of gradient projections, average projection angle
    /// - LwF: Distillation loss, temperature used
    /// </para>
    /// </remarks>
    public IReadOnlyDictionary<string, object>? StrategyMetrics { get; init; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ContinualLearningResult{T}"/> class.
    /// </summary>
    public ContinualLearningResult(
        int taskId,
        T trainingLoss,
        T trainingAccuracy,
        T averagePreviousTaskAccuracy,
        TimeSpan trainingTime,
        Vector<T> lossHistory,
        Vector<T> regularizationLossHistory)
    {
        if (taskId < 0)
            throw new ArgumentOutOfRangeException(nameof(taskId), "Task ID must be non-negative");
        if (lossHistory == null)
            throw new ArgumentNullException(nameof(lossHistory));
        if (regularizationLossHistory == null)
            throw new ArgumentNullException(nameof(regularizationLossHistory));

        TaskId = taskId;
        TrainingLoss = trainingLoss;
        TrainingAccuracy = trainingAccuracy;
        AveragePreviousTaskAccuracy = averagePreviousTaskAccuracy;
        TrainingTime = trainingTime;
        LossHistory = lossHistory;
        RegularizationLossHistory = regularizationLossHistory;
    }

    /// <summary>
    /// Returns a string representation of the training result.
    /// </summary>
    public override string ToString()
    {
        return $"Task {TaskId}: Loss={TrainingLoss}, Accuracy={TrainingAccuracy}, " +
               $"PrevTaskAccuracy={AveragePreviousTaskAccuracy}, Time={TrainingTime.TotalSeconds:F2}s";
    }
}
