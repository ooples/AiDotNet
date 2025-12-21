namespace AiDotNet.ContinualLearning.Results;

/// <summary>
/// Result from evaluating model performance on a single task.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class captures how well the model performs on a specific task
/// at a point in time. It's used to track whether the model remembers what it learned.</para>
/// </remarks>
public class TaskEvaluationResult<T>
{
    /// <summary>
    /// Gets the task identifier (0-indexed).
    /// </summary>
    public int TaskId { get; }

    /// <summary>
    /// Gets the accuracy on this task.
    /// </summary>
    public T Accuracy { get; }

    /// <summary>
    /// Gets the loss on this task.
    /// </summary>
    public T Loss { get; }

    /// <summary>
    /// Gets the number of samples evaluated.
    /// </summary>
    public int? SampleCount { get; init; }

    /// <summary>
    /// Gets the number of correct predictions.
    /// </summary>
    public int? CorrectCount { get; init; }

    /// <summary>
    /// Gets the evaluation time.
    /// </summary>
    public TimeSpan? EvaluationTime { get; init; }

    /// <summary>
    /// Gets per-class accuracy breakdown, if available.
    /// </summary>
    public IReadOnlyDictionary<int, T>? PerClassAccuracy { get; init; }

    /// <summary>
    /// Gets the confusion matrix, if available.
    /// </summary>
    /// <remarks>
    /// <para>Key format: (predicted, actual) -> count</para>
    /// </remarks>
    public IReadOnlyDictionary<(int Predicted, int Actual), int>? ConfusionMatrix { get; init; }

    /// <summary>
    /// Gets the confidence scores for predictions, if available.
    /// </summary>
    public Vector<T>? ConfidenceScores { get; init; }

    /// <summary>
    /// Gets additional metrics like F1-score, precision, recall.
    /// </summary>
    public IReadOnlyDictionary<string, T>? AdditionalMetrics { get; init; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TaskEvaluationResult{T}"/> class.
    /// </summary>
    public TaskEvaluationResult(int taskId, T accuracy, T loss)
    {
        if (taskId < 0)
            throw new ArgumentOutOfRangeException(nameof(taskId), "Task ID must be non-negative");

        TaskId = taskId;
        Accuracy = accuracy;
        Loss = loss;
    }

    /// <summary>
    /// Returns a string representation of the evaluation result.
    /// </summary>
    public override string ToString()
    {
        return $"Task {TaskId}: Accuracy={Accuracy}, Loss={Loss}";
    }
}
