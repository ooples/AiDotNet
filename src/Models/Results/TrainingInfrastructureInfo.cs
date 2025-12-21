using AiDotNet.CheckpointManagement;
using AiDotNet.ExperimentTracking;
using AiDotNet.TrainingMonitoring;
using AiDotNet.TrainingMonitoring.ExperimentTracking;

namespace AiDotNet.Models.Results;

/// <summary>
/// Contains structured experiment tracking information from a trained model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This record provides type-safe access to experiment tracking data, including
/// the experiment and run identifiers, training metrics history, and hyperparameters.
/// </para>
/// <para><b>For Beginners:</b> This is a container for all experiment-related data.
///
/// It includes:
/// - Experiment and run IDs for finding this specific training session
/// - Access to the experiment tracker for comparing runs
/// - Training metrics history for visualization
/// - Hyperparameters used during training
/// - Data version information for reproducibility
/// </para>
/// </remarks>
public record ExperimentInfo<T>(
    /// <summary>
    /// The experiment ID that groups related training runs together.
    /// </summary>
    string? ExperimentId,

    /// <summary>
    /// The unique run ID for this specific training session.
    /// </summary>
    string? RunId,

    /// <summary>
    /// The experiment run object for logging additional metrics post-training.
    /// </summary>
    IExperimentRun<T>? ExperimentRun,

    /// <summary>
    /// The experiment tracker for comparing runs and starting new experiments.
    /// </summary>
    IExperimentTracker<T>? ExperimentTracker,

    /// <summary>
    /// Training metrics history (e.g., loss, accuracy) over epochs/steps.
    /// </summary>
    Dictionary<string, List<double>>? MetricsHistory,

    /// <summary>
    /// Hyperparameters used during training.
    /// </summary>
    Dictionary<string, object>? Hyperparameters,

    /// <summary>
    /// The hyperparameter optimization trial ID, if optimization was used.
    /// </summary>
    int? TrialId,

    /// <summary>
    /// Hash of the training data for version tracking and reproducibility.
    /// </summary>
    string? DataVersionHash
);

/// <summary>
/// Contains structured model registry information from a trained model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output predictions.</typeparam>
/// <remarks>
/// <para>
/// This record provides type-safe access to model registry data, including
/// the registered model name, version, and access to the registry for lifecycle management.
/// </para>
/// <para><b>For Beginners:</b> This is a container for model versioning and registry data.
///
/// It includes:
/// - The registered model name and version number
/// - Access to the registry for stage transitions (Staging -> Production)
/// - Checkpoint path for loading/saving the model
/// - Access to the checkpoint manager for persistence operations
/// </para>
/// </remarks>
public record ModelRegistryInfo<T, TInput, TOutput>(
    /// <summary>
    /// The name under which this model is registered in the registry.
    /// </summary>
    string? RegisteredName,

    /// <summary>
    /// The version number of this model in the registry.
    /// </summary>
    int? Version,

    /// <summary>
    /// The model registry for version and lifecycle management.
    /// </summary>
    IModelRegistry<T, TInput, TOutput>? Registry,

    /// <summary>
    /// The path where the model checkpoint is stored.
    /// </summary>
    string? CheckpointPath,

    /// <summary>
    /// The checkpoint manager for saving and loading model states.
    /// </summary>
    ICheckpointManager<T, TInput, TOutput>? CheckpointManager
);
