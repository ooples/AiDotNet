namespace AiDotNet.Serving.Models;

/// <summary>
/// Request to apply LoRA fine-tuning to a loaded model.
/// </summary>
/// <remarks>
/// <para>
/// LoRA (Low-Rank Adaptation) enables efficient fine-tuning by learning low-rank
/// decompositions of weight updates instead of modifying all weights directly.
/// </para>
/// <para><b>For Beginners:</b> LoRA lets you customize a pre-trained model for your
/// specific use case with much less memory and compute than traditional fine-tuning.
/// Instead of updating all model weights, it adds small "adapter" layers that learn
/// the adjustments needed for your task.
/// </para>
/// </remarks>
public class LoRAFineTuneRequest
{
    /// <summary>
    /// Gets or sets the name of the model to fine-tune.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the training data features.
    /// Each row is a training example.
    /// </summary>
    public double[][] TrainingFeatures { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Gets or sets the training data labels/targets.
    /// Each row corresponds to the features at the same index.
    /// </summary>
    public double[][] TrainingLabels { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Gets or sets the rank of the low-rank decomposition.
    /// Lower values use fewer parameters but may be less expressive.
    /// Default is 8.
    /// </summary>
    public int Rank { get; set; } = 8;

    /// <summary>
    /// Gets or sets the scaling factor (alpha) for LoRA.
    /// The actual contribution is scaled by alpha/rank.
    /// Default is 8.0 (same as rank for 1.0 scaling).
    /// </summary>
    public double Alpha { get; set; } = 8.0;

    /// <summary>
    /// Gets or sets whether to freeze the base model weights during training.
    /// Default is true (recommended for LoRA).
    /// </summary>
    public bool FreezeBaseModel { get; set; } = true;

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// Default is 1e-4.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// Default is 3.
    /// </summary>
    public int Epochs { get; set; } = 3;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// Default is 32.
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether to save the fine-tuned model.
    /// If true, SavePath must be provided.
    /// </summary>
    public bool SaveModel { get; set; } = false;

    /// <summary>
    /// Gets or sets the path to save the fine-tuned model.
    /// Only used when SaveModel is true.
    /// </summary>
    public string? SavePath { get; set; }

    /// <summary>
    /// Gets or sets an optional request ID for tracking purposes.
    /// </summary>
    public string? RequestId { get; set; }
}

