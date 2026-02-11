using AiDotNet.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Contains the results of a training run, including the trained model, loss history, and metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> After training completes, this object tells you everything about how it went:
/// the trained model ready for predictions, the loss values at each epoch (so you can see if training
/// improved over time), how long it took, and whether it completed successfully.
/// </para>
/// </remarks>
internal class TrainingResult<T>
{
    /// <summary>
    /// Gets or sets the trained model ready for making predictions.
    /// </summary>
    public IFullModel<T, Matrix<T>, Vector<T>>? TrainedModel { get; set; }

    /// <summary>
    /// Gets or sets the loss value recorded at the end of each epoch.
    /// </summary>
    public List<T> EpochLosses { get; set; } = new();

    /// <summary>
    /// Gets or sets the total number of epochs that were executed.
    /// </summary>
    public int TotalEpochs { get; set; }

    /// <summary>
    /// Gets or sets the total wall-clock time spent training.
    /// </summary>
    public TimeSpan TrainingDuration { get; set; }

    /// <summary>
    /// Gets or sets whether the training run completed without error.
    /// True when training finishes normally (all epochs or early stopping), false on failure.
    /// </summary>
    public bool Completed { get; set; }
}
