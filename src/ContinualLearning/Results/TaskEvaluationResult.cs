namespace AiDotNet.ContinualLearning.Results;

/// <summary>
/// Result of evaluating performance on a single task.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TaskEvaluationResult<T>
{
    /// <summary>
    /// The task ID that was evaluated.
    /// </summary>
    public int TaskId { get; }

    /// <summary>
    /// Test accuracy on the task.
    /// </summary>
    public T Accuracy { get; }

    /// <summary>
    /// Test loss on the task.
    /// </summary>
    public T Loss { get; }

    /// <summary>
    /// Initializes a new instance of the TaskEvaluationResult class.
    /// </summary>
    public TaskEvaluationResult(int taskId, T accuracy, T loss)
    {
        TaskId = taskId;
        Accuracy = accuracy;
        Loss = loss;
    }
}
