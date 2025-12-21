#if !NET6_0_OR_GREATER
#pragma warning disable CS8600, CS8601, CS8602, CS8603, CS8604
#endif
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Newtonsoft.Json;
#if !NET6_0_OR_GREATER
using AiDotNet.TrainingMonitoring;
#endif

namespace AiDotNet.TrainingMonitoring.ExperimentTracking;

/// <summary>
/// Local file-based model registry for managing model versions and deployments.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> ModelRegistry provides a centralized place to store,
/// version, and manage your trained models. It tracks:
/// - Different versions of your models
/// - Which version is in production
/// - How models were created (lineage)
/// - Deployment history
///
/// Example usage:
/// <code>
/// // Create registry
/// var registry = new ModelRegistry("./model_registry");
///
/// // Register a new model
/// registry.CreateRegisteredModel("fraud-detector", "Detects fraudulent transactions");
///
/// // Create a version from a trained model
/// var version = registry.CreateModelVersion(
///     "fraud-detector",
///     "./trained_model",
///     runId: "run_123");
///
/// // Promote to production
/// registry.TransitionModelVersionStage("fraud-detector", 1, ModelStage.Production);
///
/// // Record deployment
/// registry.RecordDeployment("fraud-detector", 1, new DeploymentInfo
/// {
///     Target = "kubernetes-cluster",
///     Status = DeploymentStatus.Succeeded,
///     EndpointUrl = "https://api.example.com/predict"
/// });
/// </code>
/// </remarks>
public class ModelRegistry : IModelRegistry
{
    private readonly ConcurrentDictionary<string, RegisteredModel> _models = new();
    private readonly ConcurrentDictionary<string, ConcurrentDictionary<int, ModelVersion>> _versions = new();
    private readonly ConcurrentDictionary<string, ModelLineage> _lineages = new();
    private readonly ConcurrentDictionary<string, List<DeploymentInfo>> _deployments = new();
    private readonly object _fileLock = new();
    private bool _disposed;

    /// <inheritdoc />
    public string RegistryUri { get; }

    private string ModelsPath => Path.Combine(RegistryUri, "models");

    /// <summary>
    /// Creates a new model registry.
    /// </summary>
    /// <param name="registryUri">Base directory for the registry.</param>
    public ModelRegistry(string? registryUri = null)
    {
        RegistryUri = registryUri ?? Path.Combine(Environment.CurrentDirectory, "model_registry");
        Directory.CreateDirectory(ModelsPath);
        LoadExistingModels();
    }

    /// <inheritdoc />
    public RegisteredModel CreateRegisteredModel(string name, string? description = null, Dictionary<string, string>? tags = null)
    {
        if (_models.TryGetValue(name, out var existing))
        {
            return existing;
        }

        var modelDir = Path.Combine(ModelsPath, SanitizeName(name));
        Directory.CreateDirectory(modelDir);
        Directory.CreateDirectory(Path.Combine(modelDir, "versions"));

        var model = new RegisteredModel
        {
            Name = name,
            Description = description,
            Tags = tags ?? new Dictionary<string, string>()
        };

        _models[name] = model;
        _versions[name] = new ConcurrentDictionary<int, ModelVersion>();
        SaveModelMetadata(model);

        return model;
    }

    /// <inheritdoc />
    public RegisteredModel? GetRegisteredModel(string name)
    {
        _models.TryGetValue(name, out var model);
        return model?.IsDeleted == true ? null : model;
    }

    /// <inheritdoc />
    public List<RegisteredModel> ListRegisteredModels(string? filter = null, string? orderBy = null, int maxResults = 100)
    {
        var query = _models.Values.Where(m => !m.IsDeleted).AsEnumerable();

        var filterValue = filter;
        if (!string.IsNullOrEmpty(filterValue))
        {
            // Simple filter: "name LIKE 'pattern'"
            if (filterValue.StartsWith("name LIKE", StringComparison.OrdinalIgnoreCase))
            {
                var pattern = filterValue.Substring(10).Trim().Trim('\'', '"').Replace("%", "");
                query = query.Where(m => m.Name.Contains(pattern, StringComparison.OrdinalIgnoreCase));
            }
        }

        var orderByValue = orderBy;
        if (!string.IsNullOrEmpty(orderByValue))
        {
            query = orderByValue.ToLowerInvariant() switch
            {
                "name" => query.OrderBy(m => m.Name),
                "name desc" => query.OrderByDescending(m => m.Name),
                "created_at" => query.OrderBy(m => m.CreatedAt),
                "created_at desc" => query.OrderByDescending(m => m.CreatedAt),
                "last_updated_at" => query.OrderBy(m => m.LastUpdatedAt),
                "last_updated_at desc" => query.OrderByDescending(m => m.LastUpdatedAt),
                _ => query.OrderByDescending(m => m.LastUpdatedAt)
            };
        }
        else
        {
            query = query.OrderByDescending(m => m.LastUpdatedAt);
        }

        return query.Take(maxResults).ToList();
    }

    /// <inheritdoc />
    public void UpdateRegisteredModel(string name, string? description)
    {
        if (!_models.TryGetValue(name, out var model))
            throw new KeyNotFoundException($"Model '{name}' not found.");

        model.Description = description;
        model.LastUpdatedAt = DateTime.UtcNow;
        SaveModelMetadata(model);
    }

    /// <inheritdoc />
    public void DeleteRegisteredModel(string name)
    {
        if (_models.TryGetValue(name, out var model))
        {
            model.IsDeleted = true;
            SaveModelMetadata(model);
        }
    }

    /// <inheritdoc />
    public ModelVersion CreateModelVersion(
        string modelName,
        string sourcePath,
        string? runId = null,
        string? description = null,
        Dictionary<string, string>? tags = null)
    {
        // Ensure model exists
        if (!_models.ContainsKey(modelName))
        {
            CreateRegisteredModel(modelName);
        }

        var model = _models[modelName];
        var nextVersion = model.LatestVersion + 1;

        var versionDir = Path.Combine(ModelsPath, SanitizeName(modelName), "versions", nextVersion.ToString());
        Directory.CreateDirectory(versionDir);

        // Copy model artifacts
        var artifactDir = Path.Combine(versionDir, "artifacts");
        Directory.CreateDirectory(artifactDir);

        if (File.Exists(sourcePath))
        {
            File.Copy(sourcePath, Path.Combine(artifactDir, Path.GetFileName(sourcePath)), overwrite: true);
        }
        else if (Directory.Exists(sourcePath))
        {
            CopyDirectory(sourcePath, artifactDir);
        }

        var version = new ModelVersion
        {
            ModelName = modelName,
            Version = nextVersion,
            Description = description,
            SourcePath = artifactDir,
            RunId = runId,
            Tags = tags ?? new Dictionary<string, string>(),
            CreatedBy = Environment.UserName,
            Metadata = CollectModelMetadata(artifactDir)
        };

        var versions = _versions.GetOrAdd(modelName, _ => new ConcurrentDictionary<int, ModelVersion>());
        versions[nextVersion] = version;

        model.LatestVersion = nextVersion;
        model.LastUpdatedAt = DateTime.UtcNow;

        SaveVersionMetadata(version);
        SaveModelMetadata(model);

        return version;
    }

    /// <inheritdoc />
    public ModelVersion? GetModelVersion(string modelName, int version)
    {
        if (_versions.TryGetValue(modelName, out var versions) &&
            versions.TryGetValue(version, out var modelVersion))
        {
            return modelVersion.Status == ModelVersionStatus.Deleted ? null : modelVersion;
        }
        return null;
    }

    /// <inheritdoc />
    public ModelVersion? GetLatestVersion(string modelName, params ModelStage[] stages)
    {
        if (!_versions.TryGetValue(modelName, out var versions))
            return null;

        var query = versions.Values
            .Where(v => v.Status != ModelVersionStatus.Deleted)
            .AsEnumerable();

        if (stages.Length > 0)
        {
            var stageSet = stages.ToHashSet();
            query = query.Where(v => stageSet.Contains(v.Stage));
        }

        return query.OrderByDescending(v => v.Version).FirstOrDefault();
    }

    /// <inheritdoc />
    public List<ModelVersion> ListModelVersions(string modelName, params ModelStage[] stages)
    {
        if (!_versions.TryGetValue(modelName, out var versions))
            return new List<ModelVersion>();

        var query = versions.Values
            .Where(v => v.Status != ModelVersionStatus.Deleted)
            .AsEnumerable();

        if (stages.Length > 0)
        {
            var stageSet = stages.ToHashSet();
            query = query.Where(v => stageSet.Contains(v.Stage));
        }

        return query.OrderByDescending(v => v.Version).ToList();
    }

    /// <inheritdoc />
    public void UpdateModelVersion(string modelName, int version, string? description)
    {
        var modelVersion = GetModelVersion(modelName, version);
        if (modelVersion is null)
            throw new KeyNotFoundException($"Model version {modelName}/v{version} not found.");

        modelVersion.Description = description;
        modelVersion.LastUpdatedAt = DateTime.UtcNow;
        SaveVersionMetadata(modelVersion);
    }

    /// <inheritdoc />
    public void TransitionModelVersionStage(string modelName, int version, ModelStage stage, bool archiveExisting = false)
    {
        var modelVersion = GetModelVersion(modelName, version);
        if (modelVersion is null)
            throw new KeyNotFoundException($"Model version {modelName}/v{version} not found.");

        // Archive existing versions in target stage if requested
        if (archiveExisting && stage != ModelStage.Archived)
        {
            var existingInStage = ListModelVersions(modelName, stage);
            foreach (var existing in existingInStage)
            {
                existing.Stage = ModelStage.Archived;
                existing.LastUpdatedAt = DateTime.UtcNow;
                SaveVersionMetadata(existing);
            }
        }

        modelVersion.Stage = stage;
        modelVersion.LastUpdatedAt = DateTime.UtcNow;
        SaveVersionMetadata(modelVersion);
    }

    /// <inheritdoc />
    public void DeleteModelVersion(string modelName, int version)
    {
        var modelVersion = GetModelVersion(modelName, version);
        if (modelVersion is not null)
        {
            modelVersion.Status = ModelVersionStatus.Deleted;
            SaveVersionMetadata(modelVersion);
        }
    }

    /// <inheritdoc />
    public void SetModelVersionTag(string modelName, int version, string key, string value)
    {
        var modelVersion = GetModelVersion(modelName, version);
        if (modelVersion is null)
            throw new KeyNotFoundException($"Model version {modelName}/v{version} not found.");

        modelVersion.Tags[key] = value;
        modelVersion.LastUpdatedAt = DateTime.UtcNow;
        SaveVersionMetadata(modelVersion);
    }

    /// <inheritdoc />
    public void DeleteModelVersionTag(string modelName, int version, string key)
    {
        var modelVersion = GetModelVersion(modelName, version);
        if (modelVersion is null)
            throw new KeyNotFoundException($"Model version {modelName}/v{version} not found.");

        modelVersion.Tags.Remove(key);
        modelVersion.LastUpdatedAt = DateTime.UtcNow;
        SaveVersionMetadata(modelVersion);
    }

    /// <inheritdoc />
    public void RecordModelLineage(string modelName, int version, ModelLineage lineage)
    {
        var key = $"{modelName}/{version}";
        _lineages[key] = lineage;

        var versionDir = GetVersionDir(modelName, version);
        var lineagePath = Path.Combine(versionDir, "lineage.json");

        lock (_fileLock)
        {
            File.WriteAllText(lineagePath, JsonConvert.SerializeObject(lineage, Formatting.Indented));
        }
    }

    /// <inheritdoc />
    public ModelLineage? GetModelLineage(string modelName, int version)
    {
        var key = $"{modelName}/{version}";
        if (_lineages.TryGetValue(key, out var lineage))
            return lineage;

        var versionDir = GetVersionDir(modelName, version);
        var lineagePath = Path.Combine(versionDir, "lineage.json");

        if (File.Exists(lineagePath))
        {
            var json = File.ReadAllText(lineagePath);
            lineage = JsonConvert.DeserializeObject<ModelLineage>(json);
            if (lineage is not null)
            {
                _lineages[key] = lineage;
            }
            return lineage;
        }

        return null;
    }

    /// <inheritdoc />
    public void RecordDeployment(string modelName, int version, DeploymentInfo deployment)
    {
        deployment.ModelName = modelName;
        deployment.ModelVersion = version;

        var key = $"{modelName}/{version}";
        var deployments = _deployments.GetOrAdd(key, _ => new List<DeploymentInfo>());
        lock (deployments)
        {
            deployments.Add(deployment);
        }

        var versionDir = GetVersionDir(modelName, version);
        var deploymentsPath = Path.Combine(versionDir, "deployments.json");

        lock (_fileLock)
        {
            File.WriteAllText(deploymentsPath, JsonConvert.SerializeObject(deployments, Formatting.Indented));
        }
    }

    /// <inheritdoc />
    public List<DeploymentInfo> GetDeploymentHistory(string modelName, int version)
    {
        var key = $"{modelName}/{version}";
        if (_deployments.TryGetValue(key, out var deployments))
            return deployments.ToList();

        var versionDir = GetVersionDir(modelName, version);
        var deploymentsPath = Path.Combine(versionDir, "deployments.json");

        if (File.Exists(deploymentsPath))
        {
            var json = File.ReadAllText(deploymentsPath);
            var loaded = JsonConvert.DeserializeObject<List<DeploymentInfo>>(json);
            if (loaded is not null)
            {
                _deployments[key] = loaded;
                return loaded;
            }
        }

        return new List<DeploymentInfo>();
    }

    /// <inheritdoc />
    public DeploymentInfo? GetCurrentDeployment(string modelName)
    {
        // Find the production version
        var prodVersion = GetLatestVersion(modelName, ModelStage.Production);
        if (prodVersion is null) return null;

        var history = GetDeploymentHistory(modelName, prodVersion.Version);
        return history
            .Where(d => d.Status == DeploymentStatus.Succeeded)
            .OrderByDescending(d => d.StartedAt)
            .FirstOrDefault();
    }

    /// <inheritdoc />
    public string LoadModelArtifacts(string modelUri)
    {
        // Parse URI: "models:/model-name/version" or "models:/model-name/stage"
        if (!modelUri.StartsWith("models:/"))
            throw new ArgumentException("Invalid model URI. Expected format: models:/model-name/version or models:/model-name/stage");

        var parts = modelUri.Substring(8).Split('/');
        if (parts.Length != 2)
            throw new ArgumentException("Invalid model URI format.");

        var modelName = parts[0];
        var versionOrStage = parts[1];

        ModelVersion? version;

        if (int.TryParse(versionOrStage, out var versionNum))
        {
            version = GetModelVersion(modelName, versionNum);
        }
        else if (Enum.TryParse<ModelStage>(versionOrStage, true, out var stage))
        {
            version = GetLatestVersion(modelName, stage);
        }
        else
        {
            throw new ArgumentException($"Invalid version or stage: {versionOrStage}");
        }

        if (version is null)
            throw new KeyNotFoundException($"Model version not found: {modelUri}");

        return version.SourcePath;
    }

    /// <inheritdoc />
    public List<ModelVersion> SearchModelVersions(string? filter = null, int maxResults = 100)
    {
        var allVersions = _versions.Values
            .SelectMany(v => v.Values)
            .Where(v => v.Status != ModelVersionStatus.Deleted);

        var filterValue = filter;
        if (!string.IsNullOrEmpty(filterValue))
        {
            // Parse simple filters like "run_id = 'xxx'" or "tags.key = 'value'"
            var parts = filterValue.Split(new[] { ' ' }, 3);
            if (parts.Length >= 3)
            {
                var field = parts[0];
                var value = parts[2].Trim('\'', '"');

                if (field == "run_id")
                {
                    allVersions = allVersions.Where(v => v.RunId == value);
                }
                else if (field.StartsWith("tags."))
                {
                    var tagKey = field.Substring(5);
                    allVersions = allVersions.Where(v => v.Tags.TryGetValue(tagKey, out var tagValue) && tagValue == value);
                }
                else if (field == "stage")
                {
                    if (Enum.TryParse<ModelStage>(value, true, out var stage))
                    {
                        allVersions = allVersions.Where(v => v.Stage == stage);
                    }
                }
            }
        }

        return allVersions
            .OrderByDescending(v => v.CreatedAt)
            .Take(maxResults)
            .ToList();
    }

    private void LoadExistingModels()
    {
        if (!Directory.Exists(ModelsPath))
            return;

        foreach (var modelDir in Directory.GetDirectories(ModelsPath))
        {
            var metaPath = Path.Combine(modelDir, "metadata.json");
            if (File.Exists(metaPath))
            {
                try
                {
                    var json = File.ReadAllText(metaPath);
                    var model = JsonConvert.DeserializeObject<RegisteredModel>(json);
                    if (model is not null)
                    {
                        _models[model.Name] = model;
                        _versions[model.Name] = new ConcurrentDictionary<int, ModelVersion>();

                        // Load versions
                        var versionsDir = Path.Combine(modelDir, "versions");
                        if (Directory.Exists(versionsDir))
                        {
                            foreach (var versionDir in Directory.GetDirectories(versionsDir))
                            {
                                var versionMetaPath = Path.Combine(versionDir, "version.json");
                                if (File.Exists(versionMetaPath))
                                {
                                    var versionJson = File.ReadAllText(versionMetaPath);
                                    var version = JsonConvert.DeserializeObject<ModelVersion>(versionJson);
                                    if (version is not null)
                                    {
                                        _versions[model.Name][version.Version] = version;
                                    }
                                }
                            }
                        }
                    }
                }
                catch (Exception)
                {
                    // Skip invalid models
                }
            }
        }
    }

    private void SaveModelMetadata(RegisteredModel model)
    {
        var modelDir = Path.Combine(ModelsPath, SanitizeName(model.Name));
        Directory.CreateDirectory(modelDir);
        var metaPath = Path.Combine(modelDir, "metadata.json");

        lock (_fileLock)
        {
            File.WriteAllText(metaPath, JsonConvert.SerializeObject(model, Formatting.Indented));
        }
    }

    private void SaveVersionMetadata(ModelVersion version)
    {
        var versionDir = GetVersionDir(version.ModelName, version.Version);
        Directory.CreateDirectory(versionDir);
        var metaPath = Path.Combine(versionDir, "version.json");

        lock (_fileLock)
        {
            File.WriteAllText(metaPath, JsonConvert.SerializeObject(version, Formatting.Indented));
        }
    }

    private string GetVersionDir(string modelName, int version)
    {
        return Path.Combine(ModelsPath, SanitizeName(modelName), "versions", version.ToString());
    }

    private static string SanitizeName(string name)
    {
        foreach (var c in Path.GetInvalidFileNameChars())
        {
            name = name.Replace(c, '_');
        }
        return name;
    }

    private static void CopyDirectory(string sourceDir, string destDir)
    {
        Directory.CreateDirectory(destDir);

        foreach (var file in Directory.GetFiles(sourceDir))
        {
            var destFile = Path.Combine(destDir, Path.GetFileName(file));
            File.Copy(file, destFile, overwrite: true);
        }

        foreach (var subDir in Directory.GetDirectories(sourceDir))
        {
            var destSubDir = Path.Combine(destDir, Path.GetFileName(subDir));
            CopyDirectory(subDir, destSubDir);
        }
    }

    private static Dictionary<string, object> CollectModelMetadata(string artifactDir)
    {
        var metadata = new Dictionary<string, object>();

        // Collect basic file info
        if (Directory.Exists(artifactDir))
        {
            var files = Directory.GetFiles(artifactDir, "*", SearchOption.AllDirectories);
            metadata["file_count"] = files.Length;
            metadata["total_size_bytes"] = files.Sum(f => new FileInfo(f).Length);

            var extensions = files
                .Select(f => Path.GetExtension(f).ToLowerInvariant())
                .Where(e => !string.IsNullOrEmpty(e))
                .Distinct()
                .ToList();
            metadata["file_types"] = extensions;
        }

        // Collect environment info
        metadata["created_on"] = Environment.MachineName;
        metadata["runtime"] = RuntimeInformation.FrameworkDescription;
        metadata["os"] = RuntimeInformation.OSDescription;

        return metadata;
    }

    /// <summary>
    /// Disposes the registry.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }
}
