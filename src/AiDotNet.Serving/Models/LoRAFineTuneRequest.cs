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

    internal string? Validate()
    {
        if (string.IsNullOrWhiteSpace(ModelName))
        {
            return "ModelName is required";
        }

        if (TrainingFeatures == null || TrainingFeatures.Length == 0)
        {
            return "TrainingFeatures array is required and cannot be empty";
        }

        if (TrainingLabels == null || TrainingLabels.Length == 0)
        {
            return "TrainingLabels array is required and cannot be empty";
        }

        if (TrainingFeatures.Length != TrainingLabels.Length)
        {
            return "TrainingFeatures and TrainingLabels must have the same length";
        }

        int featureDim = TrainingFeatures[0]?.Length ?? 0;
        if (featureDim <= 0)
        {
            return "TrainingFeatures rows must be non-empty";
        }

        for (int i = 0; i < TrainingFeatures.Length; i++)
        {
            if (TrainingFeatures[i] == null || TrainingFeatures[i].Length != featureDim)
            {
                return "All TrainingFeatures rows must have the same length";
            }
        }

        int labelDim = TrainingLabels[0]?.Length ?? 0;
        if (labelDim <= 0)
        {
            return "TrainingLabels rows must be non-empty";
        }

        for (int i = 0; i < TrainingLabels.Length; i++)
        {
            if (TrainingLabels[i] == null || TrainingLabels[i].Length != labelDim)
            {
                return "All TrainingLabels rows must have the same length";
            }
        }

        if (Rank <= 0)
        {
            return "Rank must be greater than 0";
        }

        if (Alpha <= 0.0)
        {
            return "Alpha must be greater than 0";
        }

        if (LearningRate <= 0.0)
        {
            return "LearningRate must be greater than 0";
        }

        if (Epochs <= 0)
        {
            return "Epochs must be greater than 0";
        }

        if (BatchSize <= 0)
        {
            return "BatchSize must be greater than 0";
        }

        if (SaveModel && string.IsNullOrWhiteSpace(SavePath))
        {
            return "SavePath is required when SaveModel is true";
        }

        return null;
    }
}

