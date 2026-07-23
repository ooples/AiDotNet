using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Result of executing a training stage.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class TrainingStageResult<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the name of the completed stage.
    /// </summary>
    public string StageName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the model after this stage.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? Model { get; set; }

    /// <summary>
    /// Gets or sets the training metrics from this stage.
    /// </summary>
    public FineTuningMetrics<T>? Metrics { get; set; }

    /// <summary>
    /// Gets or sets evaluation metrics if evaluation was performed.
    /// </summary>
    public Dictionary<string, double>? EvaluationMetrics { get; set; }

    /// <summary>
    /// Gets or sets the duration of this stage.
    /// </summary>
    public TimeSpan Duration { get; set; }

    /// <summary>
    /// Gets or sets whether the stage completed successfully.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets any error message if the stage failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets the checkpoint path if one was saved.
    /// </summary>
    public string? CheckpointPath { get; set; }

    /// <summary>
    /// Gets or sets the stage index in the pipeline.
    /// </summary>
    public int StageIndex { get; set; }
}
