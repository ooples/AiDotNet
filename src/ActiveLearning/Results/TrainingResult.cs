using AiDotNet.LinearAlgebra;

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
    /// <param name="finalLoss">The final training loss.</param>
    /// <param name="finalAccuracy">The final training accuracy.</param>
    /// <param name="trainingTime">The time taken for training.</param>
    /// <param name="lossHistory">The history of loss values during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when lossHistory is null.</exception>
    public TrainingResult(T finalLoss, T finalAccuracy, TimeSpan trainingTime, Vector<T> lossHistory)
    {
        if (lossHistory == null)
            throw new ArgumentNullException(nameof(lossHistory));

        FinalLoss = finalLoss;
        FinalAccuracy = finalAccuracy;
        TrainingTime = trainingTime;
        LossHistory = lossHistory;
    }
}
