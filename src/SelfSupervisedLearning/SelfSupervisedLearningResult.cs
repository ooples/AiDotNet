using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Evaluation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Result from SSL pretraining containing the trained encoder and metrics.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> After SSL pretraining, this result object contains
/// everything you need: the trained encoder, training history, and evaluation metrics.
/// You can use the encoder for downstream tasks or continue training.</para>
/// </remarks>
public class SelfSupervisedLearningResult<T>
{
    /// <summary>
    /// Gets or sets the pretrained encoder network.
    /// </summary>
    /// <remarks>
    /// This is the main output of SSL pretraining. Use this encoder's representations
    /// for downstream tasks like classification, detection, or segmentation.
    /// </remarks>
    public INeuralNetwork<T>? Encoder { get; set; }

    /// <summary>
    /// Gets or sets the name of the SSL method that was used for training.
    /// </summary>
    /// <remarks>
    /// <para>Taken from <see cref="ISelfSupervisedLearningMethod{T}.Name"/> of the method that actually ran, so it
    /// reports custom methods as faithfully as shipped ones.</para>
    /// <para>Examples: "SimCLR", "MoCo v2", "BYOL", "DINO", "MAE".</para>
    /// </remarks>
    public string MethodName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the training configuration used.
    /// </summary>
    public SelfSupervisedLearningConfig<T>? Config { get; set; }

    /// <summary>
    /// Gets or sets the training history.
    /// </summary>
    public SelfSupervisedLearningTrainingHistory<T>? History { get; set; }

    /// <summary>
    /// Gets or sets the final SSL metrics.
    /// </summary>
    public SelfSupervisedLearningMetricReport<T>? FinalMetrics { get; set; }

    /// <summary>
    /// Gets or sets the linear evaluation result (if performed).
    /// </summary>
    public LinearEvalResult<T>? LinearEvaluation { get; set; }

    /// <summary>
    /// Gets or sets the k-NN evaluation accuracy (if performed).
    /// </summary>
    public double? KNNAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the total training time in seconds.
    /// </summary>
    public double TrainingTimeSeconds { get; set; }

    /// <summary>
    /// Gets or sets the number of epochs trained.
    /// </summary>
    public int EpochsTrained { get; set; }

    /// <summary>
    /// Gets or sets whether training was successful.
    /// </summary>
    public bool IsSuccess { get; set; }

    /// <summary>
    /// Gets or sets any error message if training failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets the path to saved checkpoint (if saved).
    /// </summary>
    public string? CheckpointPath { get; set; }

    /// <summary>
    /// Gets the best validation metric achieved during training.
    /// </summary>
    public T? BestValidationMetric { get; set; }

    /// <summary>
    /// Gets the epoch at which the best validation metric was achieved.
    /// </summary>
    public int BestEpoch { get; set; }

    /// <summary>
    /// Creates a successful SSL result.
    /// </summary>
    public static SelfSupervisedLearningResult<T> Success(
        INeuralNetwork<T> encoder,
        string methodName,
        SelfSupervisedLearningConfig<T> config,
        SelfSupervisedLearningTrainingHistory<T> history)
    {
        return new SelfSupervisedLearningResult<T>
        {
            Encoder = encoder,
            MethodName = methodName,
            Config = config,
            History = history,
            IsSuccess = true,
            EpochsTrained = history.LossHistory.Count
        };
    }

    /// <summary>
    /// Creates a failed SSL result.
    /// </summary>
    public static SelfSupervisedLearningResult<T> Failure(string errorMessage)
    {
        return new SelfSupervisedLearningResult<T>
        {
            IsSuccess = false,
            ErrorMessage = errorMessage
        };
    }
}
