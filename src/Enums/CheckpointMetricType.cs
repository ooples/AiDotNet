namespace AiDotNet.Enums;

/// <summary>
/// Standard metrics for checkpoint selection and early stopping.
/// </summary>
/// <remarks>
/// <para>
/// These metrics determine which checkpoint is considered "best" during training.
/// </para>
/// <para><b>For Beginners:</b> Use Loss for most cases, or Accuracy for classification.
/// The metric determines when to save checkpoints and when to stop early.</para>
/// </remarks>
public enum CheckpointMetricType
{
    /// <summary>
    /// Training or validation loss (lower is better).
    /// </summary>
    /// <remarks>
    /// The most common metric for checkpoint selection.
    /// Works for all model types and training objectives.
    /// </remarks>
    Loss = 0,

    /// <summary>
    /// Accuracy metric (higher is better).
    /// </summary>
    /// <remarks>
    /// Percentage of correct predictions.
    /// Best for classification tasks with balanced classes.
    /// </remarks>
    Accuracy = 1,

    /// <summary>
    /// Perplexity (lower is better).
    /// </summary>
    /// <remarks>
    /// Exponential of cross-entropy loss.
    /// Standard metric for language models.
    /// </remarks>
    Perplexity = 2,

    /// <summary>
    /// F1 score (higher is better).
    /// </summary>
    /// <remarks>
    /// Harmonic mean of precision and recall.
    /// Good for imbalanced classification.
    /// </remarks>
    F1Score = 3,

    /// <summary>
    /// BLEU score (higher is better).
    /// </summary>
    /// <remarks>
    /// Bilingual Evaluation Understudy score.
    /// Standard metric for machine translation and text generation.
    /// </remarks>
    BLEU = 4,

    /// <summary>
    /// ROUGE score (higher is better).
    /// </summary>
    /// <remarks>
    /// Recall-Oriented Understudy for Gisting Evaluation.
    /// Common for summarization tasks.
    /// </remarks>
    ROUGE = 5,

    /// <summary>
    /// Reward model score (higher is better).
    /// </summary>
    /// <remarks>
    /// Score from a reward model during RLHF.
    /// Used for preference optimization evaluation.
    /// </remarks>
    RewardScore = 6,

    /// <summary>
    /// Win rate against reference (higher is better).
    /// </summary>
    /// <remarks>
    /// Percentage of times model output is preferred over reference.
    /// Used for evaluating alignment quality.
    /// </remarks>
    WinRate = 7,

    /// <summary>
    /// Custom metric defined by user.
    /// </summary>
    /// <remarks>
    /// Use with CustomMetricName property to specify the metric.
    /// </remarks>
    Custom = 8
}
