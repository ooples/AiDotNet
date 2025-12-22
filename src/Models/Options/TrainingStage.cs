using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Represents a single stage in a training pipeline.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class TrainingStage<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the name of this stage for logging and identification.
    /// </summary>
    public string Name { get; set; } = "Training Stage";

    /// <summary>
    /// Gets or sets the type of training stage.
    /// </summary>
    public TrainingStageType StageType { get; set; } = TrainingStageType.SupervisedFineTuning;

    /// <summary>
    /// Gets or sets the fine-tuning method to use in this stage.
    /// </summary>
    public FineTuningMethodType FineTuningMethod { get; set; } = FineTuningMethodType.SFT;

    /// <summary>
    /// Gets or sets the training data for this stage.
    /// </summary>
    public FineTuningData<T, TInput, TOutput>? TrainingData { get; set; }

    /// <summary>
    /// Gets or sets the validation data for this stage.
    /// </summary>
    public FineTuningData<T, TInput, TOutput>? ValidationData { get; set; }

    /// <summary>
    /// Gets or sets the fine-tuning options for this stage.
    /// </summary>
    public FineTuningOptions<T>? Options { get; set; }

    /// <summary>
    /// Gets or sets the optimizer type override for this stage.
    /// </summary>
    public OptimizerType? OptimizerOverride { get; set; }

    /// <summary>
    /// Gets or sets the learning rate override for this stage.
    /// </summary>
    public double? LearningRateOverride { get; set; }

    /// <summary>
    /// Gets or sets whether this stage is evaluation-only (no training).
    /// </summary>
    public bool IsEvaluationOnly { get; set; }

    /// <summary>
    /// Gets or sets whether to freeze the base model during this stage.
    /// </summary>
    public bool FreezeBaseModel { get; set; }

    /// <summary>
    /// Gets or sets layer names/patterns to freeze during this stage.
    /// </summary>
    public string[]? FrozenLayers { get; set; }

    /// <summary>
    /// Gets or sets whether to use a reference model for preference methods.
    /// </summary>
    public bool UseReferenceModel { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to save a checkpoint after this stage.
    /// </summary>
    public bool SaveCheckpointAfter { get; set; } = true;

    /// <summary>
    /// Gets or sets the custom training function for custom stages.
    /// </summary>
    public Func<IFullModel<T, TInput, TOutput>, FineTuningData<T, TInput, TOutput>, CancellationToken, Task<IFullModel<T, TInput, TOutput>>>? CustomTrainingFunction { get; set; }

    /// <summary>
    /// Gets or sets the custom evaluation function for this stage.
    /// </summary>
    public Func<IFullModel<T, TInput, TOutput>, FineTuningData<T, TInput, TOutput>?, Task<Dictionary<string, double>>>? CustomEvaluationFunction { get; set; }

    /// <summary>
    /// Gets or sets stage-specific callbacks.
    /// </summary>
    public StageCallbacks<T, TInput, TOutput>? Callbacks { get; set; }

    /// <summary>
    /// Gets or sets early stopping configuration specific to this stage.
    /// </summary>
    public EarlyStoppingConfig? EarlyStopping { get; set; }

    /// <summary>
    /// Gets or sets the maximum duration for this stage.
    /// </summary>
    public TimeSpan? MaxDuration { get; set; }

    /// <summary>
    /// Gets or sets whether this stage is enabled (skipped if false).
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets conditions that must be met to run this stage.
    /// </summary>
    public Func<TrainingStageResult<T, TInput, TOutput>?, bool>? RunCondition { get; set; }
}
