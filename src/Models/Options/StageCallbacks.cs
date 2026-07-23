namespace AiDotNet.Models.Options;

/// <summary>
/// Callbacks for training stage events.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class StageCallbacks<T, TInput, TOutput>
{
    /// <summary>
    /// Called before the stage starts.
    /// </summary>
    public Action<TrainingStage<T, TInput, TOutput>>? OnStageStart { get; set; }

    /// <summary>
    /// Called after each epoch within the stage.
    /// </summary>
    public Action<int, double>? OnEpochComplete { get; set; }

    /// <summary>
    /// Called after each batch within the stage.
    /// </summary>
    public Action<int, double>? OnBatchComplete { get; set; }

    /// <summary>
    /// Called when the stage completes successfully.
    /// </summary>
    public Action<TrainingStageResult<T, TInput, TOutput>>? OnStageComplete { get; set; }

    /// <summary>
    /// Called if the stage encounters an error.
    /// </summary>
    public Action<Exception>? OnStageError { get; set; }
}
