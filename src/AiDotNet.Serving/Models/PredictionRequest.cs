namespace AiDotNet.Serving.Models;

/// <summary>
/// Represents a prediction request containing input features.
/// </summary>
public class PredictionRequest
{
    /// <summary>
    /// Gets or sets the input features for prediction.
    /// Can be a single array for one prediction or a 2D array for batch predictions.
    /// </summary>
    public double[][] Features { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Gets or sets an optional request ID for tracking purposes.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Represents a prediction response containing the model's predictions.
/// </summary>
public class PredictionResponse
{
    /// <summary>
    /// Gets or sets the predictions.
    /// Each array represents the output for one input sample.
    /// </summary>
    public double[][] Predictions { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Gets or sets the request ID that was provided in the request.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Gets or sets the time taken to process the request in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the batch size used for this prediction.
    /// </summary>
    public int BatchSize { get; set; }
}

/// <summary>
/// Request for text generation with speculative decoding.
/// </summary>
/// <remarks>
/// <para>
/// Speculative decoding accelerates text generation by using a smaller draft model
/// to generate candidate tokens that are then verified by the target model.
/// </para>
/// <para><b>For Beginners:</b> Think of speculative decoding like having a fast assistant
/// who suggests multiple words at once, which you then verify. Instead of generating
/// one token at a time (slow), we generate several candidates quickly and verify them
/// in parallel (fast).
/// </para>
/// </remarks>
public class SpeculativeDecodingRequest
{
    /// <summary>
    /// Gets or sets the input token IDs to continue from.
    /// </summary>
    public int[] InputTokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the maximum number of new tokens to generate.
    /// </summary>
    public int MaxNewTokens { get; set; } = 100;

    /// <summary>
    /// Gets or sets the sampling temperature. Higher values make output more random.
    /// Default is 1.0.
    /// </summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the end-of-sequence token ID. Generation stops when this token is produced.
    /// </summary>
    public int? EosTokenId { get; set; }

    /// <summary>
    /// Gets or sets the number of draft tokens to generate per verification step.
    /// Default is 5.
    /// </summary>
    public int NumDraftTokens { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use tree-based speculation for higher acceptance rates.
    /// Default is false.
    /// </summary>
    public bool UseTreeSpeculation { get; set; } = false;

    /// <summary>
    /// Gets or sets the branching factor for tree speculation.
    /// Only used when UseTreeSpeculation is true. Default is 2.
    /// </summary>
    public int TreeBranchFactor { get; set; } = 2;

    /// <summary>
    /// Gets or sets the maximum tree depth for tree speculation.
    /// Only used when UseTreeSpeculation is true. Default is 4.
    /// </summary>
    public int MaxTreeDepth { get; set; } = 4;

    /// <summary>
    /// Gets or sets an optional request ID for tracking purposes.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Response from text generation with speculative decoding.
/// </summary>
public class SpeculativeDecodingResponse
{
    /// <summary>
    /// Gets or sets all tokens including input and generated tokens.
    /// </summary>
    public int[] AllTokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets only the newly generated tokens.
    /// </summary>
    public int[] GeneratedTokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the number of tokens generated.
    /// </summary>
    public int NumGenerated { get; set; }

    /// <summary>
    /// Gets or sets the acceptance rate (ratio of draft tokens accepted by target model).
    /// </summary>
    public double AcceptanceRate { get; set; }

    /// <summary>
    /// Gets or sets the time taken to process the request in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the request ID that was provided in the request.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Gets or sets any error message if generation failed.
    /// </summary>
    public string? Error { get; set; }
}

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
