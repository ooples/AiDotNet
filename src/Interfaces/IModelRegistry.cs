namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for model registry systems that manage trained model storage and versioning.
/// </summary>
/// <remarks>
/// A model registry serves as a centralized repository for trained models, managing their lifecycle
/// from development through production deployment.
///
/// <b>For Beginners:</b> Think of a model registry like a library for your trained models.
/// Just like a library catalogs books and tracks which are checked out, a model registry:
/// - Stores all your trained models in one place
/// - Tracks different versions of the same model
/// - Records which models are being used in production
/// - Helps you find and retrieve the right model when you need it
///
/// Key features include:
/// - Model versioning (keeping track of model evolution)
/// - Metadata tracking (when trained, by whom, with what data)
/// - Stage management (development, staging, production)
/// - Model comparison and lineage tracking
///
/// Why model registries matter:
/// - Prevents losing track of trained models
/// - Enables rollback to previous versions if needed
/// - Provides audit trail for compliance
/// - Facilitates collaboration between team members
/// - Simplifies deployment process
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IModelRegistry<T, TInput, TOutput>
{
    /// <summary>
    /// Registers a new model in the registry.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This adds a trained model to the registry with all its information,
    /// like adding a new book to a library catalog.
    /// </remarks>
    /// <param name="name">The name for this model.</param>
    /// <param name="model">The trained model to register.</param>
    /// <param name="metadata">Metadata about the model (training data, hyperparameters, etc.).</param>
    /// <param name="tags">Tags for categorizing the model.</param>
    /// <returns>The unique identifier for the registered model.</returns>
    string RegisterModel<TMetadata>(
        string name,
        IModel<TInput, TOutput, TMetadata> model,
        ModelMetadata<T> metadata,
        Dictionary<string, string>? tags = null) where TMetadata : class;

    /// <summary>
    /// Creates a new version of an existing model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This saves a new version of a model you've already registered,
    /// like publishing a new edition of a book.
    /// </remarks>
    /// <param name="modelName">The name of the model to version.</param>
    /// <param name="model">The new version of the model.</param>
    /// <param name="metadata">Metadata for this version.</param>
    /// <param name="description">Description of changes in this version.</param>
    /// <returns>The version number assigned.</returns>
    int CreateModelVersion<TMetadata>(
        string modelName,
        IModel<TInput, TOutput, TMetadata> model,
        ModelMetadata<T> metadata,
        string? description = null) where TMetadata : class;

    /// <summary>
    /// Retrieves a specific model version from the registry.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version number to retrieve. If null, gets latest.</param>
    /// <returns>The registered model version.</returns>
    RegisteredModel<T, TInput, TOutput> GetModel(string modelName, int? version = null);

    /// <summary>
    /// Gets the latest version of a model.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <returns>The latest version of the model.</returns>
    RegisteredModel<T, TInput, TOutput> GetLatestModel(string modelName);

    /// <summary>
    /// Gets the model currently in a specific stage (e.g., production).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Models go through stages like Development → Staging → Production.
    /// This gets the model currently in a specific stage.
    /// </remarks>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="stage">The stage to get the model from.</param>
    /// <returns>The model in that stage, or null if none.</returns>
    RegisteredModel<T, TInput, TOutput>? GetModelByStage(string modelName, ModelStage stage);

    /// <summary>
    /// Transitions a model version to a different stage.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This moves a model between stages, like promoting a model
    /// from Staging to Production after testing it.
    /// </remarks>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version to transition.</param>
    /// <param name="targetStage">The stage to transition to.</param>
    /// <param name="archivePrevious">Whether to archive the previous model in that stage.</param>
    void TransitionModelStage(string modelName, int version, ModelStage targetStage, bool archivePrevious = true);

    /// <summary>
    /// Lists all models in the registry.
    /// </summary>
    /// <param name="filter">Optional filter expression.</param>
    /// <param name="tags">Optional tags to filter by.</param>
    /// <returns>List of model names matching the criteria.</returns>
    List<string> ListModels(string? filter = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Lists all versions of a specific model.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <returns>List of all versions with metadata.</returns>
    List<ModelVersionInfo<T>> ListModelVersions(string modelName);

    /// <summary>
    /// Searches for models based on criteria.
    /// </summary>
    /// <param name="searchCriteria">Search criteria (name patterns, tags, metrics, etc.).</param>
    /// <returns>List of models matching the criteria.</returns>
    List<RegisteredModel<T, TInput, TOutput>> SearchModels(ModelSearchCriteria<T> searchCriteria);

    /// <summary>
    /// Updates the metadata for a model version.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version to update.</param>
    /// <param name="metadata">New metadata.</param>
    void UpdateModelMetadata(string modelName, int version, ModelMetadata<T> metadata);

    /// <summary>
    /// Adds or updates tags for a model.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version to tag.</param>
    /// <param name="tags">Tags to add or update.</param>
    void UpdateModelTags(string modelName, int version, Dictionary<string, string> tags);

    /// <summary>
    /// Deletes a specific model version.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version to delete.</param>
    void DeleteModelVersion(string modelName, int version);

    /// <summary>
    /// Deletes all versions of a model.
    /// </summary>
    /// <param name="modelName">The name of the model to delete.</param>
    void DeleteModel(string modelName);

    /// <summary>
    /// Compares two model versions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This shows you the differences between two versions of a model,
    /// helping you understand what changed and which performs better.
    /// </remarks>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version1">First version to compare.</param>
    /// <param name="version2">Second version to compare.</param>
    /// <returns>Comparison results showing differences.</returns>
    ModelComparison<T> CompareModels(string modelName, int version1, int version2);

    /// <summary>
    /// Gets the lineage information for a model (how it was created).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Lineage shows the "family tree" of a model - what data was used,
    /// what experiment it came from, and how it's related to other models.
    /// </remarks>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version to get lineage for.</param>
    /// <returns>Lineage information.</returns>
    ModelLineage GetModelLineage(string modelName, int version);

    /// <summary>
    /// Archives a model version.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Archiving makes a model read-only and marks it as no longer
    /// active, but keeps it available for reference.
    /// </remarks>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version to archive.</param>
    void ArchiveModel(string modelName, int version);

    /// <summary>
    /// Gets the storage location for a model version.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version number.</param>
    /// <returns>File path or URI where the model is stored.</returns>
    string GetModelStoragePath(string modelName, int version);
}

/// <summary>
/// Represents the stages a model can be in during its lifecycle.
/// </summary>
public enum ModelStage
{
    /// <summary>Model is being developed and tested.</summary>
    Development,
    /// <summary>Model is in staging for pre-production testing.</summary>
    Staging,
    /// <summary>Model is deployed in production.</summary>
    Production,
    /// <summary>Model has been archived and is no longer active.</summary>
    Archived
}
