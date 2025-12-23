using AiDotNet.AdversarialRobustness.Documentation;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Newtonsoft.Json;

namespace AiDotNet.ModelRegistry;

/// <summary>
/// Implementation of model registry for managing trained model storage and versioning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This is a complete implementation of a model registry that manages
/// the lifecycle of your trained models.
///
/// Features include:
/// - Model versioning (track different versions of the same model)
/// - Lifecycle stages (Development, Staging, Production, Archived)
/// - Model comparison and lineage tracking
/// - Persistent storage with JSON serialization
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public class ModelRegistry<T, TInput, TOutput> : ModelRegistryBase<T, TInput, TOutput>
{
    private readonly Dictionary<string, List<RegisteredModel<T, TInput, TOutput>>> _models;
    private readonly Dictionary<string, ModelLineage> _lineage;

    /// <summary>
    /// Initializes a new instance of the ModelRegistry class.
    /// </summary>
    /// <param name="registryDirectory">Directory to store models. Defaults to "./model_registry".</param>
    public ModelRegistry(string? registryDirectory = null) : base(registryDirectory)
    {
        _models = new Dictionary<string, List<RegisteredModel<T, TInput, TOutput>>>();
        _lineage = new Dictionary<string, ModelLineage>();

        LoadExistingModels();
    }

    /// <summary>
    /// Registers a new model in the registry.
    /// </summary>
    public override string RegisterModel<TMetadata>(
        string name,
        IModel<TInput, TOutput, TMetadata> model,
        ModelMetadata<T> metadata,
        Dictionary<string, string>? tags = null)
    {
        ValidateModelName(name);

        if (model == null)
            throw new ArgumentNullException(nameof(model));

        lock (SyncLock)
        {
            // Create model directory if needed
            var modelDir = GetModelDirectoryPath(name);
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            var version = 1;
            if (_models.ContainsKey(name))
            {
                version = _models[name].Max(m => m.Version) + 1;
            }
            else
            {
                _models[name] = new List<RegisteredModel<T, TInput, TOutput>>();
            }

            var registeredModel = new RegisteredModel<T, TInput, TOutput>
            {
                ModelId = Guid.NewGuid().ToString("N"),
                Name = name,
                Version = version,
                Stage = ModelStage.Development,
                Metadata = metadata,
                Tags = tags ?? new Dictionary<string, string>(),
                CreatedAt = DateTime.UtcNow,
                LastModifiedAt = DateTime.UtcNow,
                StoragePath = GetModelVersionPath(name, version)
            };

            _models[name].Add(registeredModel);
            SaveModelVersion(registeredModel);

            return registeredModel.ModelId;
        }
    }

    /// <summary>
    /// Creates a new version of an existing model.
    /// </summary>
    public override int CreateModelVersion<TMetadata>(
        string modelName,
        IModel<TInput, TOutput, TMetadata> model,
        ModelMetadata<T> metadata,
        string? description = null)
    {
        ValidateModelName(modelName);

        if (model == null)
            throw new ArgumentNullException(nameof(model));

        lock (SyncLock)
        {
            if (!_models.ContainsKey(modelName))
                throw new ArgumentException($"Model '{modelName}' not found in registry.", nameof(modelName));

            var version = _models[modelName].Max(m => m.Version) + 1;

            var registeredModel = new RegisteredModel<T, TInput, TOutput>
            {
                ModelId = Guid.NewGuid().ToString("N"),
                Name = modelName,
                Version = version,
                Stage = ModelStage.Development,
                Metadata = metadata,
                Description = description,
                CreatedAt = DateTime.UtcNow,
                LastModifiedAt = DateTime.UtcNow,
                StoragePath = GetModelVersionPath(modelName, version)
            };

            _models[modelName].Add(registeredModel);
            SaveModelVersion(registeredModel);

            return version;
        }
    }

    /// <summary>
    /// Retrieves a specific model version from the registry.
    /// </summary>
    public override RegisteredModel<T, TInput, TOutput> GetModel(string modelName, int? version = null)
    {
        ValidateModelName(modelName);

        lock (SyncLock)
        {
            if (!_models.TryGetValue(modelName, out var versions))
                throw new ArgumentException($"Model '{modelName}' not found in registry.", nameof(modelName));

            if (version.HasValue)
            {
                var model = versions.FirstOrDefault(m => m.Version == version.Value);
                if (model == null)
                    throw new ArgumentException($"Version {version.Value} of model '{modelName}' not found.", nameof(version));
                return model;
            }

            return versions.OrderByDescending(m => m.Version).First();
        }
    }

    /// <summary>
    /// Gets the latest version of a model.
    /// </summary>
    public override RegisteredModel<T, TInput, TOutput> GetLatestModel(string modelName)
    {
        return GetModel(modelName, null);
    }

    /// <summary>
    /// Gets the model currently in a specific stage.
    /// </summary>
    public override RegisteredModel<T, TInput, TOutput>? GetModelByStage(string modelName, ModelStage stage)
    {
        ValidateModelName(modelName);

        lock (SyncLock)
        {
            if (!_models.TryGetValue(modelName, out var versions))
                return null;

            return versions
                .Where(m => m.Stage == stage)
                .OrderByDescending(m => m.Version)
                .FirstOrDefault();
        }
    }

    /// <summary>
    /// Transitions a model version to a different stage.
    /// </summary>
    public override void TransitionModelStage(string modelName, int version, ModelStage targetStage, bool archivePrevious = true)
    {
        ValidateModelName(modelName);

        lock (SyncLock)
        {
            var model = GetModel(modelName, version);

            if (archivePrevious && targetStage != ModelStage.Archived)
            {
                // Archive any existing model in the target stage
                var existingInStage = _models[modelName]
                    .Where(m => m.Stage == targetStage && m.Version != version)
                    .ToList();

                foreach (var existing in existingInStage)
                {
                    existing.Stage = ModelStage.Archived;
                    existing.LastModifiedAt = DateTime.UtcNow;
                    SaveModelVersion(existing);
                }
            }

            model.Stage = targetStage;
            model.LastModifiedAt = DateTime.UtcNow;
            SaveModelVersion(model);
        }
    }

    /// <summary>
    /// Lists all models in the registry.
    /// </summary>
    public override List<string> ListModels(string? filter = null, Dictionary<string, string>? tags = null)
    {
        lock (SyncLock)
        {
            IEnumerable<string> modelNames = _models.Keys;

            if (!string.IsNullOrWhiteSpace(filter))
            {
                modelNames = modelNames.Where(n => n.Contains(filter, StringComparison.OrdinalIgnoreCase));
            }

            if (tags != null && tags.Count > 0)
            {
                modelNames = modelNames.Where(name =>
                {
                    var latestVersion = _models[name].OrderByDescending(m => m.Version).First();
                    return tags.All(t => latestVersion.Tags.TryGetValue(t.Key, out var value) && value == t.Value);
                });
            }

            return modelNames.OrderBy(n => n).ToList();
        }
    }

    /// <summary>
    /// Lists all versions of a specific model.
    /// </summary>
    public override List<ModelVersionInfo<T>> ListModelVersions(string modelName)
    {
        ValidateModelName(modelName);

        lock (SyncLock)
        {
            if (!_models.TryGetValue(modelName, out var versions))
                throw new ArgumentException($"Model '{modelName}' not found in registry.", nameof(modelName));

            return versions
                .OrderByDescending(m => m.Version)
                .Select(m => new ModelVersionInfo<T>
                {
                    Version = m.Version,
                    CreatedAt = m.CreatedAt,
                    Stage = m.Stage,
                    Description = m.Description,
                    Metadata = m.Metadata
                })
                .ToList();
        }
    }

    /// <summary>
    /// Searches for models based on criteria.
    /// </summary>
    public override List<RegisteredModel<T, TInput, TOutput>> SearchModels(ModelSearchCriteria<T> searchCriteria)
    {
        if (searchCriteria == null)
            throw new ArgumentNullException(nameof(searchCriteria));

        lock (SyncLock)
        {
            var results = _models.Values.SelectMany(v => v).AsEnumerable();

            if (!string.IsNullOrWhiteSpace(searchCriteria.NamePattern))
            {
                results = results.Where(m => m.Name.Contains(searchCriteria.NamePattern, StringComparison.OrdinalIgnoreCase));
            }

            if (searchCriteria.Stage.HasValue)
            {
                results = results.Where(m => m.Stage == searchCriteria.Stage.Value);
            }

            if (searchCriteria.MinVersion.HasValue)
            {
                results = results.Where(m => m.Version >= searchCriteria.MinVersion.Value);
            }

            if (searchCriteria.MaxVersion.HasValue)
            {
                results = results.Where(m => m.Version <= searchCriteria.MaxVersion.Value);
            }

            if (searchCriteria.CreatedAfter.HasValue)
            {
                results = results.Where(m => m.CreatedAt >= searchCriteria.CreatedAfter.Value);
            }

            if (searchCriteria.CreatedBefore.HasValue)
            {
                results = results.Where(m => m.CreatedAt <= searchCriteria.CreatedBefore.Value);
            }

            if (searchCriteria.Tags != null && searchCriteria.Tags.Count > 0)
            {
                results = results.Where(m =>
                    searchCriteria.Tags.All(t => m.Tags.TryGetValue(t.Key, out var value) && value == t.Value));
            }

            return results.OrderByDescending(m => m.CreatedAt).ToList();
        }
    }

    /// <summary>
    /// Updates the metadata for a model version.
    /// </summary>
    public override void UpdateModelMetadata(string modelName, int version, ModelMetadata<T> metadata)
    {
        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        lock (SyncLock)
        {
            var model = GetModel(modelName, version);
            model.Metadata = metadata;
            model.LastModifiedAt = DateTime.UtcNow;
            SaveModelVersion(model);
        }
    }

    /// <summary>
    /// Adds or updates tags for a model.
    /// </summary>
    public override void UpdateModelTags(string modelName, int version, Dictionary<string, string> tags)
    {
        if (tags == null)
            throw new ArgumentNullException(nameof(tags));

        lock (SyncLock)
        {
            var model = GetModel(modelName, version);
            foreach (var tag in tags)
            {
                model.Tags[tag.Key] = tag.Value;
            }
            model.LastModifiedAt = DateTime.UtcNow;
            SaveModelVersion(model);
        }
    }

    /// <summary>
    /// Deletes a specific model version.
    /// </summary>
    public override void DeleteModelVersion(string modelName, int version)
    {
        lock (SyncLock)
        {
            if (!_models.TryGetValue(modelName, out var versions))
                return;

            var model = versions.FirstOrDefault(m => m.Version == version);
            if (model == null)
                return;

            versions.Remove(model);

            // Delete file - validate path is within registry directory before deletion
            if (model.StoragePath != null && File.Exists(model.StoragePath))
            {
                try
                {
                    ValidatePathWithinDirectory(model.StoragePath, RegistryDirectory);
                    File.Delete(model.StoragePath);
                }
                catch (UnauthorizedAccessException)
                {
                    // StoragePath points outside the registry directory, skip deletion for security
                    // This could happen if the serialized data was tampered with
                }
            }

            // If no versions left, remove the model entirely
            if (versions.Count == 0)
            {
                _models.Remove(modelName);
                var modelDir = GetModelDirectoryPath(modelName);
                if (Directory.Exists(modelDir))
                {
                    Directory.Delete(modelDir, true);
                }
            }
        }
    }

    /// <summary>
    /// Deletes all versions of a model.
    /// </summary>
    public override void DeleteModel(string modelName)
    {
        lock (SyncLock)
        {
            if (!_models.ContainsKey(modelName))
                return;

            _models.Remove(modelName);

            var modelDir = GetModelDirectoryPath(modelName);
            if (Directory.Exists(modelDir))
            {
                Directory.Delete(modelDir, true);
            }
        }
    }

    /// <summary>
    /// Compares two model versions.
    /// </summary>
    public override ModelComparison<T> CompareModels(string modelName, int version1, int version2)
    {
        lock (SyncLock)
        {
            var model1 = GetModel(modelName, version1);
            var model2 = GetModel(modelName, version2);

            var comparison = new ModelComparison<T>
            {
                Version1 = version1,
                Version2 = version2
            };

            // Compare metadata properties
            if (model1.Metadata != null && model2.Metadata != null)
            {
                if (model1.Metadata.FeatureCount != model2.Metadata.FeatureCount)
                {
                    comparison.MetadataDifferences["FeatureCount"] = (model1.Metadata.FeatureCount, model2.Metadata.FeatureCount);
                    comparison.ArchitectureChanged = true;
                }

                if (model1.Metadata.Complexity != model2.Metadata.Complexity)
                {
                    comparison.MetadataDifferences["Complexity"] = (model1.Metadata.Complexity, model2.Metadata.Complexity);
                }

                if (model1.Metadata.ModelType != model2.Metadata.ModelType)
                {
                    comparison.MetadataDifferences["ModelType"] = (model1.Metadata.ModelType, model2.Metadata.ModelType);
                    comparison.ArchitectureChanged = true;
                }

                // Compare feature importance
                var allFeatures = model1.Metadata.FeatureImportance.Keys
                    .Union(model2.Metadata.FeatureImportance.Keys)
                    .ToList();

                foreach (var feature in allFeatures)
                {
                    var has1 = model1.Metadata.FeatureImportance.TryGetValue(feature, out var val1);
                    var has2 = model2.Metadata.FeatureImportance.TryGetValue(feature, out var val2);

                    if (has1 && has2 && val1 is not null && val2 is not null)
                    {
                        comparison.MetricDifferences[$"FeatureImportance.{feature}"] = (val1, val2);
                    }
                }
            }

            return comparison;
        }
    }

    /// <summary>
    /// Gets the lineage information for a model.
    /// </summary>
    public override ModelLineage GetModelLineage(string modelName, int version)
    {
        var key = $"{modelName}:v{version}";

        lock (SyncLock)
        {
            if (_lineage.TryGetValue(key, out var lineage))
            {
                return lineage;
            }

            // Return default lineage if not tracked
            var model = GetModel(modelName, version);
            return new ModelLineage
            {
                ModelName = modelName,
                Version = version,
                CreatedAt = model.CreatedAt
            };
        }
    }

    /// <summary>
    /// Archives a model version.
    /// </summary>
    public override void ArchiveModel(string modelName, int version)
    {
        TransitionModelStage(modelName, version, ModelStage.Archived, false);
    }

    /// <summary>
    /// Gets the storage location for a model version.
    /// </summary>
    public override string GetModelStoragePath(string modelName, int version)
    {
        return GetModelVersionPath(modelName, version);
    }

    /// <summary>
    /// Attaches a Model Card to a registered model version.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A Model Card is like a nutrition label for AI models.
    /// This method associates a Model Card with a specific version of your model,
    /// documenting its intended use, limitations, and performance characteristics.</para>
    /// </remarks>
    public override void AttachModelCard(string modelName, int version, ModelCard modelCard)
    {
        if (modelCard == null)
            throw new ArgumentNullException(nameof(modelCard));

        lock (SyncLock)
        {
            var model = GetModel(modelName, version);
            model.ModelCard = modelCard;
            model.LastModifiedAt = DateTime.UtcNow;
            SaveModelVersion(model);

            // Also save the Model Card as a separate file for easy access
            var modelCardPath = GetModelCardPath(modelName, version);
            ValidatePathWithinDirectory(modelCardPath, RegistryDirectory);
            var json = SerializeToJson(modelCard);
            File.WriteAllText(modelCardPath, json);
        }
    }

    /// <summary>
    /// Gets the Model Card for a registered model version.
    /// </summary>
    public override ModelCard? GetModelCard(string modelName, int version)
    {
        lock (SyncLock)
        {
            var model = GetModel(modelName, version);
            if (model.ModelCard != null)
            {
                return model.ModelCard;
            }

            // Try to load from file if not in memory
            var modelCardPath = GetModelCardPath(modelName, version);
            if (File.Exists(modelCardPath))
            {
                try
                {
                    ValidatePathWithinDirectory(modelCardPath, RegistryDirectory);
                    var json = File.ReadAllText(modelCardPath);
                    var modelCard = DeserializeFromJson<ModelCard>(json);
                    if (modelCard != null)
                    {
                        model.ModelCard = modelCard;
                    }
                    return modelCard;
                }
                catch (IOException)
                {
                    return null;
                }
            }

            return null;
        }
    }

    /// <summary>
    /// Generates a Model Card from the registered model's metadata and evaluation results.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This automatically creates a Model Card from information
    /// already stored about your model. It's a quick way to create documentation without
    /// having to fill in everything manually.</para>
    /// </remarks>
    public override ModelCard GenerateModelCard(string modelName, int version, string? developers = null)
    {
        lock (SyncLock)
        {
            var model = GetModel(modelName, version);
            var numOps = MathHelper.GetNumericOperations<T>();

            var modelCard = new ModelCard
            {
                ModelName = modelName,
                Version = $"{version}.0.0",
                Date = model.CreatedAt,
                Developers = developers ?? string.Empty,
                ModelType = model.Metadata?.ModelType.ToString() ?? "Unknown"
            };

            // Extract performance metrics from metadata
            if (model.Metadata?.FeatureImportance != null)
            {
                var performanceMetrics = new Dictionary<string, double>();
                foreach (var kvp in model.Metadata.FeatureImportance)
                {
                    performanceMetrics[$"FeatureImportance_{kvp.Key}"] = numOps.ToDouble(kvp.Value);
                }

                if (performanceMetrics.Count > 0)
                {
                    modelCard.PerformanceMetrics["FeatureImportance"] = performanceMetrics;
                }
            }

            // Add metadata as additional info
            if (model.Metadata != null)
            {
                modelCard.TrainingData = $"Feature count: {model.Metadata.FeatureCount}, Complexity: {model.Metadata.Complexity}";
            }

            // Add tags as recommendations if relevant
            if (model.Tags != null)
            {
                foreach (var tag in model.Tags)
                {
                    if (tag.Key.StartsWith("recommendation_", StringComparison.OrdinalIgnoreCase))
                    {
                        modelCard.Recommendations.Add(tag.Value);
                    }
                    else if (tag.Key.StartsWith("limitation_", StringComparison.OrdinalIgnoreCase))
                    {
                        modelCard.Limitations.Add(tag.Value);
                    }
                }
            }

            // Add standard recommendations
            modelCard.Recommendations.Add("Continuously monitor model performance in production");
            modelCard.Recommendations.Add("Regularly evaluate model for fairness across demographic groups");
            modelCard.Recommendations.Add("Update the model with new data as needed");

            // Add standard caveats
            modelCard.Caveats.Add("Model performance may degrade over time due to data drift");
            modelCard.Caveats.Add("Results should be validated in your specific use case before deployment");

            return modelCard;
        }
    }

    /// <summary>
    /// Saves the Model Card for a model version to a file.
    /// </summary>
    /// <remarks>
    /// The filePath must resolve to a location within the registry directory
    /// to prevent path traversal attacks.
    /// </remarks>
    public override void SaveModelCard(string modelName, int version, string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        // Validate path is within registry directory to prevent path traversal
        var resolvedPath = Path.GetFullPath(filePath);
        ValidatePathWithinDirectory(resolvedPath, RegistryDirectory);

        var modelCard = GetModelCard(modelName, version);
        if (modelCard == null)
        {
            // Generate one if it doesn't exist
            modelCard = GenerateModelCard(modelName, version);
        }

        modelCard.SaveToFile(resolvedPath);
    }

    private string GetModelCardPath(string modelName, int version)
    {
        var modelDir = GetModelDirectoryPath(modelName);
        var path = Path.Combine(modelDir, $"v{version}_modelcard.json");
        ValidatePathWithinDirectory(path, RegistryDirectory);
        return path;
    }

    #region Private Helper Methods

    private void LoadExistingModels()
    {
        if (!Directory.Exists(RegistryDirectory))
            return;

        foreach (var modelDir in Directory.GetDirectories(RegistryDirectory))
        {
            var modelName = Path.GetFileName(modelDir);
            var versionFiles = Directory.GetFiles(modelDir, "v*.json");

            foreach (var versionFile in versionFiles)
            {
                try
                {
                    ValidatePathWithinDirectory(versionFile, RegistryDirectory);
                    var json = File.ReadAllText(versionFile);
                    var model = DeserializeFromJson<RegisteredModel<T, TInput, TOutput>>(json);

                    if (model != null)
                    {
                        if (!_models.ContainsKey(modelName))
                        {
                            _models[modelName] = new List<RegisteredModel<T, TInput, TOutput>>();
                        }
                        _models[modelName].Add(model);
                    }
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"[ModelRegistry] Failed to read model file '{versionFile}': {ex.Message}");
                }
                catch (JsonException ex)
                {
                    Console.WriteLine($"[ModelRegistry] Failed to deserialize model file '{versionFile}': {ex.Message}");
                }
            }
        }
    }

    private void SaveModelVersion(RegisteredModel<T, TInput, TOutput> model)
    {
        var modelDir = GetModelDirectoryPath(model.Name);
        if (!Directory.Exists(modelDir))
        {
            Directory.CreateDirectory(modelDir);
        }

        var filePath = GetModelVersionPath(model.Name, model.Version);
        ValidatePathWithinDirectory(filePath, RegistryDirectory);

        var json = SerializeToJson(model);
        File.WriteAllText(filePath, json);
    }

    #endregion
}
