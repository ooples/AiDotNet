namespace AiDotNet.ActiveLearning.Results;

/// <summary>
/// Result of training a model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TrainingResult<T>
{
    /// <summary>
    /// Final training loss.
    /// </summary>
    public T FinalLoss { get; }

    /// <summary>
    /// Final training accuracy.
    /// </summary>
    public T FinalAccuracy { get; }

    /// <summary>
    /// Training time.
    /// </summary>
    public TimeSpan TrainingTime { get; }

    /// <summary>
    /// Loss history during training.
    /// </summary>
    public Vector<T> LossHistory { get; }

    /// <summary>
    /// Initializes a new instance of the TrainingResult class.
    /// </summary>
    public TrainingResult(T finalLoss, T finalAccuracy, TimeSpan trainingTime, Vector<T> lossHistory)
    {
        FinalLoss = finalLoss;
        FinalAccuracy = finalAccuracy;
        TrainingTime = trainingTime;
        LossHistory = lossHistory;
    }
}
