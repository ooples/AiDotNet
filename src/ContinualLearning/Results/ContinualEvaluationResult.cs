using AiDotNet.LinearAlgebra;

namespace AiDotNet.ContinualLearning.Results;

/// <summary>
/// Result of evaluating performance on all learned tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class shows how well the model performs on all tasks
/// it has learned so far. It helps measure if the model has forgotten old tasks.</para>
/// </remarks>
public class ContinualEvaluationResult<T>
{
    /// <summary>
    /// Accuracy for each task (indexed by task ID).
    /// </summary>
    public Vector<T> TaskAccuracies { get; }

    /// <summary>
    /// Loss for each task (indexed by task ID).
    /// </summary>
    public Vector<T> TaskLosses { get; }

    /// <summary>
    /// Average accuracy across all tasks.
    /// </summary>
    public T AverageAccuracy { get; }

    /// <summary>
    /// Average loss across all tasks.
    /// </summary>
    public T AverageLoss { get; }

    /// <summary>
    /// Backward transfer: average change in accuracy on previous tasks after learning new tasks.
    /// Positive values indicate positive backward transfer (learning new tasks improved old ones).
    /// Negative values indicate forgetting.
    /// </summary>
    public T BackwardTransfer { get; }

    /// <summary>
    /// Forward transfer: average initial accuracy on new tasks before any training.
    /// Measures how well the model generalizes to new tasks from what it learned before.
    /// </summary>
    public T ForwardTransfer { get; }

    /// <summary>
    /// Time taken for evaluation.
    /// </summary>
    public TimeSpan EvaluationTime { get; }

    /// <summary>
    /// Initializes a new instance of the ContinualEvaluationResult class.
    /// </summary>
    public ContinualEvaluationResult(
        Vector<T> taskAccuracies,
        Vector<T> taskLosses,
        T averageAccuracy,
        T averageLoss,
        T backwardTransfer,
        T forwardTransfer,
        TimeSpan evaluationTime)
    {
        TaskAccuracies = taskAccuracies;
        TaskLosses = taskLosses;
        AverageAccuracy = averageAccuracy;
        AverageLoss = averageLoss;
        BackwardTransfer = backwardTransfer;
        ForwardTransfer = forwardTransfer;
        EvaluationTime = evaluationTime;
    }
}
