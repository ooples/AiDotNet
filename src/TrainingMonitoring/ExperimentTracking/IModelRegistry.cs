using AiDotNet.AdversarialRobustness.Documentation;

namespace AiDotNet.TrainingMonitoring.ExperimentTracking;

/// <summary>
/// Model version stage in the deployment lifecycle.
/// </summary>
public enum ModelStage
{
    /// <summary>Model is not assigned to any stage.</summary>
    None,
    /// <summary>Model is in staging/testing.</summary>
    Staging,
    /// <summary>Model is in production.</summary>
    Production,
    /// <summary>Model is archived.</summary>
    Archived
}

/// <summary>
/// Status of a model version.
/// </summary>
public enum ModelVersionStatus
{
    /// <summary>Model is pending registration.</summary>
    Pending,
    /// <summary>Model is ready for use.</summary>
    Ready,
    /// <summary>Model failed during validation.</summary>
    Failed,
    /// <summary>Model registration was deleted.</summary>
    Deleted
}

/// <summary>
/// Interface for a model registry that manages model versions and deployments.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A Model Registry is like a central library for your
/// trained models. It helps you:
/// - Store and version models
/// - Track which model is in production
/// - Manage model lifecycle (staging -> production -> archived)
/// - Record model lineage (what data/experiments created this model)
///
/// Think of it like the MLflow Model Registry - it's where your best models
/// get promoted for production use.
///
/// Key concepts:
/// - Registered Model: A named model (e.g., "fraud-detector")
/// - Model Version: A specific version of that model (e.g., v1, v2)
/// - Stage: Where the model is in its lifecycle (Staging, Production, Archived)
/// - Lineage: The experiment/run/data that created this model
/// </remarks>
public interface IModelRegistry : IDisposable
{
    /// <summary>
    /// Gets the registry URI.
    /// </summary>
    string RegistryUri { get; }

    /// <summary>
    /// Registers a new model or gets existing.
    /// </summary>
    /// <param name="name">Model name.</param>
    /// <param name="description">Optional description.</param>
    /// <param name="tags">Optional tags.</param>
    /// <returns>Registered model info.</returns>
    RegisteredModel CreateRegisteredModel(string name, string? description = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Gets a registered model by name.
    /// </summary>
    /// <param name="name">Model name.</param>
    /// <returns>Model info or null if not found.</returns>
    RegisteredModel? GetRegisteredModel(string name);

    /// <summary>
    /// Lists all registered models.
    /// </summary>
    /// <param name="filter">Optional filter (e.g., "name LIKE 'fraud%'").</param>
    /// <param name="orderBy">Optional ordering.</param>
    /// <param name="maxResults">Maximum results.</param>
    /// <returns>List of registered models.</returns>
    List<RegisteredModel> ListRegisteredModels(string? filter = null, string? orderBy = null, int maxResults = 100);

    /// <summary>
    /// Updates a registered model.
    /// </summary>
    /// <param name="name">Model name.</param>
    /// <param name="description">New description.</param>
    void UpdateRegisteredModel(string name, string? description);

    /// <summary>
    /// Deletes a registered model and all its versions.
    /// </summary>
    /// <param name="name">Model name.</param>
    void DeleteRegisteredModel(string name);

    /// <summary>
    /// Creates a new model version.
    /// </summary>
    /// <param name="modelName">Registered model name.</param>
    /// <param name="sourcePath">Path to model artifacts.</param>
    /// <param name="runId">Optional run ID that created this model.</param>
    /// <param name="description">Optional description.</param>
    /// <param name="tags">Optional tags.</param>
    /// <returns>Model version info.</returns>
    ModelVersion CreateModelVersion(
        string modelName,
        string sourcePath,
        string? runId = null,
        string? description = null,
        Dictionary<string, string>? tags = null);

    /// <summary>
    /// Gets a specific model version.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <returns>Model version or null.</returns>
    ModelVersion? GetModelVersion(string modelName, int version);

    /// <summary>
    /// Gets the latest model version.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="stages">Optional stages to filter by.</param>
    /// <returns>Latest version or null.</returns>
    ModelVersion? GetLatestVersion(string modelName, params ModelStage[] stages);

    /// <summary>
    /// Lists all versions of a model.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="stages">Optional stages to filter by.</param>
    /// <returns>List of versions.</returns>
    List<ModelVersion> ListModelVersions(string modelName, params ModelStage[] stages);

    /// <summary>
    /// Updates a model version.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <param name="description">New description.</param>
    void UpdateModelVersion(string modelName, int version, string? description);

    /// <summary>
    /// Transitions a model version to a new stage.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <param name="stage">New stage.</param>
    /// <param name="archiveExisting">Whether to archive existing versions in the target stage.</param>
    void TransitionModelVersionStage(string modelName, int version, ModelStage stage, bool archiveExisting = false);

    /// <summary>
    /// Deletes a model version.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    void DeleteModelVersion(string modelName, int version);

    /// <summary>
    /// Sets a tag on a model version.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <param name="key">Tag key.</param>
    /// <param name="value">Tag value.</param>
    void SetModelVersionTag(string modelName, int version, string key, string value);

    /// <summary>
    /// Deletes a tag from a model version.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <param name="key">Tag key.</param>
    void DeleteModelVersionTag(string modelName, int version, string key);

    /// <summary>
    /// Records model lineage information.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <param name="lineage">Lineage information.</param>
    void RecordModelLineage(string modelName, int version, ModelLineage lineage);

    /// <summary>
    /// Gets model lineage information.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <returns>Lineage info or null.</returns>
    ModelLineage? GetModelLineage(string modelName, int version);

    /// <summary>
    /// Records a deployment of a model version.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <param name="deployment">Deployment information.</param>
    void RecordDeployment(string modelName, int version, DeploymentInfo deployment);

    /// <summary>
    /// Gets deployment history for a model version.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <param name="version">Version number.</param>
    /// <returns>List of deployments.</returns>
    List<DeploymentInfo> GetDeploymentHistory(string modelName, int version);

    /// <summary>
    /// Gets the current production deployment.
    /// </summary>
    /// <param name="modelName">Model name.</param>
    /// <returns>Current deployment or null.</returns>
    DeploymentInfo? GetCurrentDeployment(string modelName);

    /// <summary>
    /// Loads a model from the registry.
    /// </summary>
    /// <param name="modelUri">URI in format "models:/model-name/version" or "models:/model-name/stage".</param>
    /// <returns>Path to the model artifacts.</returns>
    string LoadModelArtifacts(string modelUri);

    /// <summary>
    /// Searches for model versions.
    /// </summary>
    /// <param name="filter">Filter expression.</param>
    /// <param name="maxResults">Maximum results.</param>
    /// <returns>Matching versions.</returns>
    List<ModelVersion> SearchModelVersions(string? filter = null, int maxResults = 100);
}

/// <summary>
/// A registered model in the registry.
/// </summary>
public class RegisteredModel
{
    /// <summary>
    /// Gets or sets the model name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets when the model was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets when the model was last updated.
    /// </summary>
    public DateTime LastUpdatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the latest version number.
    /// </summary>
    public int LatestVersion { get; set; }

    /// <summary>
    /// Gets or sets tags.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets whether the model is deleted.
    /// </summary>
    public bool IsDeleted { get; set; }
}

/// <summary>
/// A specific version of a registered model.
/// </summary>
public class ModelVersion
{
    /// <summary>
    /// Gets or sets the model name.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version number.
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets when this version was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets when this version was last updated.
    /// </summary>
    public DateTime LastUpdatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the source path.
    /// </summary>
    public string SourcePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the run ID that created this model.
    /// </summary>
    public string? RunId { get; set; }

    /// <summary>
    /// Gets or sets the current stage.
    /// </summary>
    public ModelStage Stage { get; set; } = ModelStage.None;

    /// <summary>
    /// Gets or sets the status.
    /// </summary>
    public ModelVersionStatus Status { get; set; } = ModelVersionStatus.Ready;

    /// <summary>
    /// Gets or sets the status message.
    /// </summary>
    public string? StatusMessage { get; set; }

    /// <summary>
    /// Gets or sets tags.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets the user who created this version.
    /// </summary>
    public string? CreatedBy { get; set; }

    /// <summary>
    /// Gets or sets model metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets the Model Card for this model version.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A Model Card documents the model's intended use,
    /// limitations, performance metrics, and ethical considerations for transparent
    /// and responsible AI practices.</para>
    /// </remarks>
    public ModelCard? ModelCard { get; set; }

    /// <summary>
    /// Gets the model URI.
    /// </summary>
    public string ModelUri => $"models:/{ModelName}/{Version}";
}

/// <summary>
/// Model lineage information tracking how the model was created.
/// </summary>
public class ModelLineage
{
    /// <summary>
    /// Gets or sets the experiment ID.
    /// </summary>
    public string? ExperimentId { get; set; }

    /// <summary>
    /// Gets or sets the experiment name.
    /// </summary>
    public string? ExperimentName { get; set; }

    /// <summary>
    /// Gets or sets the run ID.
    /// </summary>
    public string? RunId { get; set; }

    /// <summary>
    /// Gets or sets the run name.
    /// </summary>
    public string? RunName { get; set; }

    /// <summary>
    /// Gets or sets the training parameters.
    /// </summary>
    public Dictionary<string, string> Parameters { get; set; } = new();

    /// <summary>
    /// Gets or sets the final metrics.
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>
    /// Gets or sets input datasets.
    /// </summary>
    public List<DatasetLineage> InputDatasets { get; set; } = new();

    /// <summary>
    /// Gets or sets the source code commit.
    /// </summary>
    public string? GitCommit { get; set; }

    /// <summary>
    /// Gets or sets the source code branch.
    /// </summary>
    public string? GitBranch { get; set; }

    /// <summary>
    /// Gets or sets environment information.
    /// </summary>
    public EnvironmentInfo? Environment { get; set; }

    /// <summary>
    /// Gets or sets the training start time.
    /// </summary>
    public DateTime? TrainingStartTime { get; set; }

    /// <summary>
    /// Gets or sets the training end time.
    /// </summary>
    public DateTime? TrainingEndTime { get; set; }

    /// <summary>
    /// Gets the training duration.
    /// </summary>
    public TimeSpan? TrainingDuration => TrainingEndTime.HasValue && TrainingStartTime.HasValue
        ? TrainingEndTime.Value - TrainingStartTime.Value
        : null;
}

/// <summary>
/// Dataset lineage information.
/// </summary>
public class DatasetLineage
{
    /// <summary>
    /// Gets or sets the dataset ID.
    /// </summary>
    public string DatasetId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the dataset name.
    /// </summary>
    public string? DatasetName { get; set; }

    /// <summary>
    /// Gets or sets the version ID.
    /// </summary>
    public string? VersionId { get; set; }

    /// <summary>
    /// Gets or sets the content hash.
    /// </summary>
    public string? ContentHash { get; set; }

    /// <summary>
    /// Gets or sets the role (train, validation, test).
    /// </summary>
    public string? Role { get; set; }
}

/// <summary>
/// Environment information for reproducibility.
/// </summary>
public class EnvironmentInfo
{
    /// <summary>
    /// Gets or sets the OS version.
    /// </summary>
    public string? OperatingSystem { get; set; }

    /// <summary>
    /// Gets or sets the .NET runtime version.
    /// </summary>
    public string? RuntimeVersion { get; set; }

    /// <summary>
    /// Gets or sets installed packages/dependencies.
    /// </summary>
    public Dictionary<string, string> Dependencies { get; set; } = new();

    /// <summary>
    /// Gets or sets hardware information.
    /// </summary>
    public HardwareInfo? Hardware { get; set; }
}

/// <summary>
/// Hardware information.
/// </summary>
public class HardwareInfo
{
    /// <summary>
    /// Gets or sets CPU information.
    /// </summary>
    public string? Cpu { get; set; }

    /// <summary>
    /// Gets or sets total memory in MB.
    /// </summary>
    public long TotalMemoryMb { get; set; }

    /// <summary>
    /// Gets or sets GPU information.
    /// </summary>
    public List<GpuInfo> Gpus { get; set; } = new();
}

/// <summary>
/// GPU information.
/// </summary>
public class GpuInfo
{
    /// <summary>
    /// Gets or sets the GPU name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the memory in MB.
    /// </summary>
    public long MemoryMb { get; set; }

    /// <summary>
    /// Gets or sets the driver version.
    /// </summary>
    public string? DriverVersion { get; set; }
}

/// <summary>
/// Deployment status.
/// </summary>
public enum DeploymentStatus
{
    /// <summary>Deployment is pending.</summary>
    Pending,
    /// <summary>Deployment is in progress.</summary>
    InProgress,
    /// <summary>Deployment succeeded.</summary>
    Succeeded,
    /// <summary>Deployment failed.</summary>
    Failed,
    /// <summary>Deployment was rolled back.</summary>
    RolledBack,
    /// <summary>Deployment is retired/replaced.</summary>
    Retired
}

/// <summary>
/// Information about a model deployment.
/// </summary>
public class DeploymentInfo
{
    /// <summary>
    /// Gets or sets the deployment ID.
    /// </summary>
    public string DeploymentId { get; set; } = Guid.NewGuid().ToString("N");

    /// <summary>
    /// Gets or sets the model name.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the model version.
    /// </summary>
    public int ModelVersion { get; set; }

    /// <summary>
    /// Gets or sets the deployment target.
    /// </summary>
    public string Target { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the deployment status.
    /// </summary>
    public DeploymentStatus Status { get; set; } = DeploymentStatus.Pending;

    /// <summary>
    /// Gets or sets the status message.
    /// </summary>
    public string? StatusMessage { get; set; }

    /// <summary>
    /// Gets or sets when the deployment started.
    /// </summary>
    public DateTime StartedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets when the deployment completed.
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the endpoint URL.
    /// </summary>
    public string? EndpointUrl { get; set; }

    /// <summary>
    /// Gets or sets who initiated the deployment.
    /// </summary>
    public string? InitiatedBy { get; set; }

    /// <summary>
    /// Gets or sets deployment configuration.
    /// </summary>
    public Dictionary<string, object> Configuration { get; set; } = new();

    /// <summary>
    /// Gets or sets metrics from the deployment.
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();
}
