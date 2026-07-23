namespace AiDotNet.Serving.Models;

/// <summary>
/// Response from LoRA fine-tuning.
/// </summary>
public class LoRAFineTuneResponse
{
    /// <summary>
    /// Gets or sets whether fine-tuning completed successfully.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the name of the fine-tuned model.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the final training loss.
    /// </summary>
    public double FinalLoss { get; set; }

    /// <summary>
    /// Gets or sets the training history (loss per epoch).
    /// </summary>
    public double[] LossHistory { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the number of trainable parameters added by LoRA.
    /// </summary>
    public long TrainableParameters { get; set; }

    /// <summary>
    /// Gets or sets the path where the model was saved.
    /// Null if SaveModel was false.
    /// </summary>
    public string? SavedPath { get; set; }

    /// <summary>
    /// Gets or sets the time taken for fine-tuning in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the request ID that was provided in the request.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Gets or sets any error message if fine-tuning failed.
    /// </summary>
    public string? Error { get; set; }
}

