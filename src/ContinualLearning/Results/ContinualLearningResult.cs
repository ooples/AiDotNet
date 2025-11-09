using AiDotNet.LinearAlgebra;

namespace AiDotNet.ContinualLearning.Results;

/// <summary>
/// Result of training on a single task in continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class contains the results after learning a new task,
/// including how well the model performs on the new task and whether it forgot previous tasks.</para>
/// </remarks>
public class ContinualLearningResult<T>
{
    /// <summary>
    /// The task ID that was learned.
    /// </summary>
    public int TaskId { get; }

    /// <summary>
    /// Final training loss on the new task.
    /// </summary>
    public T TrainingLoss { get; }

    /// <summary>
    /// Final training accuracy on the new task.
    /// </summary>
    public T TrainingAccuracy { get; }

    /// <summary>
    /// Average accuracy across all previously learned tasks (excluding the current one).
    /// </summary>
    public T AveragePreviousTaskAccuracy { get; }

    /// <summary>
    /// Time taken to learn this task.
    /// </summary>
    public TimeSpan TrainingTime { get; }

    /// <summary>
    /// Loss history during training.
    /// </summary>
    public Vector<T> LossHistory { get; }

    /// <summary>
    /// Regularization loss history (e.g., EWC penalty).
    /// </summary>
    public Vector<T> RegularizationLossHistory { get; }

    /// <summary>
    /// Initializes a new instance of the ContinualLearningResult class.
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
        TaskId = taskId;
        TrainingLoss = trainingLoss;
        TrainingAccuracy = trainingAccuracy;
        AveragePreviousTaskAccuracy = averagePreviousTaskAccuracy;
        TrainingTime = trainingTime;
        LossHistory = lossHistory;
        RegularizationLossHistory = regularizationLossHistory;
    }
}
