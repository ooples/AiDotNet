using AiDotNet.Interfaces;

namespace AiDotNet.Models;

/// <summary>
/// Represents a registered model in the model registry with its metadata and versioning information.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public class RegisteredModel<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the unique identifier for this model.
    /// </summary>
    public string ModelId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the name of the model.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version number.
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets the current stage of the model.
    /// </summary>
    public ModelStage Stage { get; set; } = ModelStage.Development;

    /// <summary>
    /// Gets or sets the model metadata.
    /// </summary>
    public ModelMetadata<T>? Metadata { get; set; }

    /// <summary>
    /// Gets or sets the tags associated with this model.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets when this model was registered.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets when this model was last modified.
    /// </summary>
    public DateTime LastModifiedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the description of this model version.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the storage path for the model.
    /// </summary>
    public string? StoragePath { get; set; }
}

/// <summary>
/// Information about a specific model version.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ModelVersionInfo<T>
{
    /// <summary>
    /// Gets or sets the version number.
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets the creation timestamp.
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Gets or sets the current stage.
    /// </summary>
    public ModelStage Stage { get; set; } = ModelStage.Development;

    /// <summary>
    /// Gets or sets the description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the model metadata.
    /// </summary>
    public ModelMetadata<T>? Metadata { get; set; }
}

/// <summary>
/// Criteria for searching models in the registry.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ModelSearchCriteria<T>
{
    /// <summary>
    /// Gets or sets the name pattern to match.
    /// </summary>
    public string? NamePattern { get; set; }

    /// <summary>
    /// Gets or sets the tags to filter by.
    /// </summary>
    public Dictionary<string, string>? Tags { get; set; }

    /// <summary>
    /// Gets or sets the stage to filter by.
    /// </summary>
    public ModelStage? Stage { get; set; }

    /// <summary>
    /// Gets or sets the minimum version.
    /// </summary>
    public int? MinVersion { get; set; }

    /// <summary>
    /// Gets or sets the maximum version.
    /// </summary>
    public int? MaxVersion { get; set; }

    /// <summary>
    /// Gets or sets the created after date filter.
    /// </summary>
    public DateTime? CreatedAfter { get; set; }

    /// <summary>
    /// Gets or sets the created before date filter.
    /// </summary>
    public DateTime? CreatedBefore { get; set; }
}

/// <summary>
/// Comparison results between two model versions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ModelComparison<T>
{
    /// <summary>
    /// Gets or sets the first model version.
    /// </summary>
    public int Version1 { get; set; }

    /// <summary>
    /// Gets or sets the second model version.
    /// </summary>
    public int Version2 { get; set; }

    /// <summary>
    /// Gets or sets the metadata differences.
    /// </summary>
    public Dictionary<string, (object? Value1, object? Value2)> MetadataDifferences { get; set; } = new();

    /// <summary>
    /// Gets or sets the metric differences.
    /// </summary>
    public Dictionary<string, (T Value1, T Value2)> MetricDifferences { get; set; } = new();

    /// <summary>
    /// Gets or sets whether the model architecture changed.
    /// </summary>
    public bool ArchitectureChanged { get; set; }
}

/// <summary>
/// Lineage information for a model showing its origin and dependencies.
/// </summary>
public class ModelLineage
{
    /// <summary>
    /// Gets or sets the model name.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version.
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets the experiment ID that produced this model.
    /// </summary>
    public string? ExperimentId { get; set; }

    /// <summary>
    /// Gets or sets the run ID that produced this model.
    /// </summary>
    public string? RunId { get; set; }

    /// <summary>
    /// Gets or sets the training data source.
    /// </summary>
    public string? TrainingDataSource { get; set; }

    /// <summary>
    /// Gets or sets parent model information (if this was derived from another model).
    /// </summary>
    public string? ParentModel { get; set; }

    /// <summary>
    /// Gets or sets the parent model version.
    /// </summary>
    public int? ParentVersion { get; set; }

    /// <summary>
    /// Gets or sets the creation timestamp.
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Gets or sets the creator/user who trained this model.
    /// </summary>
    public string? Creator { get; set; }
}
