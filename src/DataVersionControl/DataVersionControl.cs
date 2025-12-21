using AiDotNet.Interfaces;
using AiDotNet.Models;
using Newtonsoft.Json;

namespace AiDotNet.DataVersionControl;

/// <summary>
/// Implementation of data version control for tracking dataset changes over time.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This is a complete implementation of data version control that manages
/// the lifecycle of your datasets, similar to how Git manages code.
///
/// Features include:
/// - Dataset versioning with hash-based integrity verification
/// - Linking datasets to training runs for reproducibility
/// - Tagging versions for easy reference
/// - Lineage tracking for data provenance
/// - Multi-dataset snapshots for experiment reproducibility
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class DataVersionControl<T> : DataVersionControlBase<T>
{
    private readonly Dictionary<string, List<DatasetVersion<T>>> _datasets;
    private readonly Dictionary<string, DatasetLineage> _lineage;
    private readonly Dictionary<string, Dictionary<string, string>> _versionTags; // datasetName -> tag -> versionHash
    private readonly Dictionary<string, (string DatasetName, string VersionHash, string? ModelId)> _runLinks;
    private readonly Dictionary<string, MultiDatasetSnapshot> _snapshots;

    /// <summary>
    /// Initializes a new instance of the DataVersionControl class.
    /// </summary>
    /// <param name="storageDirectory">Directory to store version control data. Defaults to "./data_versions".</param>
    public DataVersionControl(string? storageDirectory = null) : base(storageDirectory)
    {
        _datasets = new Dictionary<string, List<DatasetVersion<T>>>();
        _lineage = new Dictionary<string, DatasetLineage>();
        _versionTags = new Dictionary<string, Dictionary<string, string>>();
        _runLinks = new Dictionary<string, (string, string, string?)>();
        _snapshots = new Dictionary<string, MultiDatasetSnapshot>();

        LoadExistingData();
    }

    /// <summary>
    /// Creates a new dataset version.
    /// </summary>
    public override string CreateDatasetVersion(
        string datasetName,
        string dataPath,
        string? description = null,
        Dictionary<string, object>? metadata = null,
        Dictionary<string, string>? tags = null)
    {
        ValidateDatasetName(datasetName);

        if (string.IsNullOrWhiteSpace(dataPath))
            throw new ArgumentException("Data path cannot be null or empty.", nameof(dataPath));

        lock (SyncLock)
        {
            var hash = ComputeDatasetHash(dataPath);
            var version = 1;

            if (_datasets.ContainsKey(datasetName))
            {
                version = _datasets[datasetName].Max(d => d.Version) + 1;
            }
            else
            {
                _datasets[datasetName] = new List<DatasetVersion<T>>();
                _versionTags[datasetName] = new Dictionary<string, string>();
            }

            var datasetVersion = new DatasetVersion<T>
            {
                VersionId = hash,
                DatasetName = datasetName,
                Version = version,
                Hash = hash,
                StoragePath = dataPath,
                CreatedAt = DateTime.UtcNow,
                Description = description,
                Tags = tags ?? new Dictionary<string, string>(),
                SizeBytes = GetDataSize(dataPath)
            };

            _datasets[datasetName].Add(datasetVersion);
            SaveDatasetVersion(datasetVersion);

            return hash;
        }
    }

    /// <summary>
    /// Retrieves a specific version of a dataset.
    /// </summary>
    public override DatasetVersion<T> GetDatasetVersion(string datasetName, string? versionHash = null)
    {
        ValidateDatasetName(datasetName);

        lock (SyncLock)
        {
            if (!_datasets.TryGetValue(datasetName, out var versions))
                throw new ArgumentException($"Dataset '{datasetName}' not found.", nameof(datasetName));

            if (string.IsNullOrWhiteSpace(versionHash))
            {
                return versions.OrderByDescending(v => v.Version).First();
            }

            var dataset = versions.FirstOrDefault(v => v.Hash == versionHash || v.VersionId == versionHash);
            if (dataset == null)
                throw new ArgumentException($"Version '{versionHash}' of dataset '{datasetName}' not found.", nameof(versionHash));

            return dataset;
        }
    }

    /// <summary>
    /// Gets the latest version of a dataset.
    /// </summary>
    public override DatasetVersion<T> GetLatestDatasetVersion(string datasetName)
    {
        return GetDatasetVersion(datasetName, null);
    }

    /// <summary>
    /// Lists all versions of a dataset.
    /// </summary>
    public override List<DatasetVersionInfo<T>> ListDatasetVersions(string datasetName)
    {
        ValidateDatasetName(datasetName);

        lock (SyncLock)
        {
            if (!_datasets.TryGetValue(datasetName, out var versions))
                throw new ArgumentException($"Dataset '{datasetName}' not found.", nameof(datasetName));

            return versions
                .OrderByDescending(v => v.Version)
                .Select(v => new DatasetVersionInfo<T>
                {
                    VersionId = v.VersionId,
                    Version = v.Version,
                    CreatedAt = v.CreatedAt,
                    Hash = v.Hash,
                    RecordCount = v.RecordCount,
                    SizeBytes = v.SizeBytes,
                    Description = v.Description
                })
                .ToList();
        }
    }

    /// <summary>
    /// Lists all tracked datasets.
    /// </summary>
    public override List<string> ListDatasets(string? filter = null, Dictionary<string, string>? tags = null)
    {
        lock (SyncLock)
        {
            IEnumerable<string> datasetNames = _datasets.Keys;

            if (!string.IsNullOrWhiteSpace(filter))
            {
                datasetNames = datasetNames.Where(n => n.Contains(filter, StringComparison.OrdinalIgnoreCase));
            }

            if (tags != null && tags.Count > 0)
            {
                datasetNames = datasetNames.Where(name =>
                {
                    var latestVersion = _datasets[name].OrderByDescending(d => d.Version).First();
                    return tags.All(t => latestVersion.Tags.TryGetValue(t.Key, out var value) && value == t.Value);
                });
            }

            return datasetNames.OrderBy(n => n).ToList();
        }
    }

    /// <summary>
    /// Links a dataset version to a model training run.
    /// </summary>
    public override void LinkDatasetToRun(string datasetName, string versionHash, string runId, string? modelId = null)
    {
        ValidateDatasetName(datasetName);

        if (string.IsNullOrWhiteSpace(runId))
            throw new ArgumentException("Run ID cannot be null or empty.", nameof(runId));

        lock (SyncLock)
        {
            // Verify dataset version exists
            var datasetVersion = GetDatasetVersion(datasetName, versionHash);

            _runLinks[runId] = (datasetName, versionHash, modelId);

            // Update lineage
            var lineageKey = $"{datasetName}:{versionHash}";
            if (_lineage.TryGetValue(lineageKey, out var lineage))
            {
                if (!lineage.UsedInRuns.Contains(runId))
                {
                    lineage.UsedInRuns.Add(runId);
                }
            }
            else
            {
                _lineage[lineageKey] = new DatasetLineage
                {
                    DatasetName = datasetName,
                    Version = datasetVersion.Version,
                    CreatedAt = DateTime.UtcNow,
                    UsedInRuns = new List<string> { runId }
                };
            }

            SaveRunLinks();
        }
    }

    /// <summary>
    /// Gets all training runs that used a specific dataset version.
    /// </summary>
    public override List<string> GetRunsUsingDataset(string datasetName, string versionHash)
    {
        lock (SyncLock)
        {
            return _runLinks
                .Where(kvp => kvp.Value.DatasetName == datasetName && kvp.Value.VersionHash == versionHash)
                .Select(kvp => kvp.Key)
                .ToList();
        }
    }

    /// <summary>
    /// Gets the dataset version used by a specific training run.
    /// </summary>
    public override DatasetVersion<T> GetDatasetForRun(string runId)
    {
        if (string.IsNullOrWhiteSpace(runId))
            throw new ArgumentException("Run ID cannot be null or empty.", nameof(runId));

        lock (SyncLock)
        {
            if (!_runLinks.TryGetValue(runId, out var link))
                throw new ArgumentException($"No dataset linked to run '{runId}'.", nameof(runId));

            return GetDatasetVersion(link.DatasetName, link.VersionHash);
        }
    }

    /// <summary>
    /// Tags a dataset version for easy reference.
    /// </summary>
    public override void TagDatasetVersion(string datasetName, string versionHash, string tag)
    {
        ValidateDatasetName(datasetName);

        if (string.IsNullOrWhiteSpace(tag))
            throw new ArgumentException("Tag cannot be null or empty.", nameof(tag));

        lock (SyncLock)
        {
            // Verify version exists
            GetDatasetVersion(datasetName, versionHash);

            if (!_versionTags.ContainsKey(datasetName))
            {
                _versionTags[datasetName] = new Dictionary<string, string>();
            }

            _versionTags[datasetName][tag] = versionHash;
            SaveVersionTags();
        }
    }

    /// <summary>
    /// Gets a dataset version by its tag.
    /// </summary>
    public override DatasetVersion<T> GetDatasetByTag(string datasetName, string tag)
    {
        ValidateDatasetName(datasetName);

        if (string.IsNullOrWhiteSpace(tag))
            throw new ArgumentException("Tag cannot be null or empty.", nameof(tag));

        lock (SyncLock)
        {
            if (!_versionTags.TryGetValue(datasetName, out var tags))
                throw new ArgumentException($"No tags found for dataset '{datasetName}'.", nameof(datasetName));

            if (!tags.TryGetValue(tag, out var versionHash))
                throw new ArgumentException($"Tag '{tag}' not found for dataset '{datasetName}'.", nameof(tag));

            return GetDatasetVersion(datasetName, versionHash);
        }
    }

    /// <summary>
    /// Compares two dataset versions to see what changed.
    /// </summary>
    public override DatasetComparison<T> CompareDatasetVersions(string datasetName, string version1Hash, string version2Hash)
    {
        lock (SyncLock)
        {
            var version1 = GetDatasetVersion(datasetName, version1Hash);
            var version2 = GetDatasetVersion(datasetName, version2Hash);

            var comparison = new DatasetComparison<T>
            {
                Version1 = version1.Version,
                Version2 = version2.Version
            };

            // Compare sizes
            if (version1.SizeBytes != version2.SizeBytes)
            {
                comparison.SchemaChanges.Add($"Size changed from {version1.SizeBytes} to {version2.SizeBytes} bytes");
            }

            // Compare record counts
            if (version1.RecordCount != version2.RecordCount)
            {
                var diff = version2.RecordCount - version1.RecordCount;
                if (diff > 0)
                {
                    comparison.RecordsAdded = diff;
                }
                else
                {
                    comparison.RecordsRemoved = -diff;
                }
            }

            // Check if hashes differ (indicates modification)
            if (version1.Hash != version2.Hash)
            {
                comparison.RecordsModified = Math.Max(1, Math.Min(version1.RecordCount, version2.RecordCount) / 10);
            }

            return comparison;
        }
    }

    /// <summary>
    /// Records metadata about how a dataset was created or transformed.
    /// </summary>
    public override void RecordDatasetLineage(string datasetName, string versionHash, DatasetLineage lineage)
    {
        ValidateDatasetName(datasetName);

        if (lineage == null)
            throw new ArgumentNullException(nameof(lineage));

        lock (SyncLock)
        {
            // Verify version exists
            GetDatasetVersion(datasetName, versionHash);

            var key = $"{datasetName}:{versionHash}";
            _lineage[key] = lineage;
            SaveLineage();
        }
    }

    /// <summary>
    /// Gets the lineage information for a dataset version.
    /// </summary>
    public override DatasetLineage GetDatasetLineage(string datasetName, string versionHash)
    {
        lock (SyncLock)
        {
            var key = $"{datasetName}:{versionHash}";

            if (_lineage.TryGetValue(key, out var lineage))
            {
                return lineage;
            }

            // Return default lineage
            var version = GetDatasetVersion(datasetName, versionHash);
            return new DatasetLineage
            {
                DatasetName = datasetName,
                Version = version.Version,
                CreatedAt = version.CreatedAt
            };
        }
    }

    /// <summary>
    /// Deletes a specific dataset version.
    /// </summary>
    public override void DeleteDatasetVersion(string datasetName, string versionHash)
    {
        lock (SyncLock)
        {
            if (!_datasets.TryGetValue(datasetName, out var versions))
                return;

            var version = versions.FirstOrDefault(v => v.Hash == versionHash || v.VersionId == versionHash);
            if (version == null)
                return;

            versions.Remove(version);

            // Remove associated lineage
            var lineageKey = $"{datasetName}:{versionHash}";
            _lineage.Remove(lineageKey);

            // Remove tags pointing to this version
            if (_versionTags.TryGetValue(datasetName, out var tags))
            {
                var tagsToRemove = tags.Where(t => t.Value == versionHash).Select(t => t.Key).ToList();
                foreach (var tag in tagsToRemove)
                {
                    tags.Remove(tag);
                }
            }

            // Delete version file
            var versionPath = GetVersionFilePath(datasetName, version.Version);
            if (File.Exists(versionPath))
            {
                File.Delete(versionPath);
            }

            // If no versions left, clean up dataset directory
            if (versions.Count == 0)
            {
                _datasets.Remove(datasetName);
                _versionTags.Remove(datasetName);

                var datasetDir = GetDatasetDirectoryPath(datasetName);
                if (Directory.Exists(datasetDir))
                {
                    Directory.Delete(datasetDir, true);
                }
            }
        }
    }

    /// <summary>
    /// Gets statistics about a dataset version.
    /// </summary>
    public override DatasetStatistics<T> GetDatasetStatistics(string datasetName, string versionHash)
    {
        lock (SyncLock)
        {
            var version = GetDatasetVersion(datasetName, versionHash);

            // Return basic statistics based on stored metadata
            return new DatasetStatistics<T>
            {
                RecordCount = version.RecordCount,
                ColumnCount = 0, // Would need to parse the data to determine
                MissingValues = new Dictionary<string, long>(),
                NumericStats = new Dictionary<string, NumericColumnStats<T>>(),
                CategoricalStats = new Dictionary<string, CategoricalColumnStats>()
            };
        }
    }

    /// <summary>
    /// Creates a snapshot of multiple related datasets together.
    /// </summary>
    public override string CreateDatasetSnapshot(
        string snapshotName,
        Dictionary<string, string> datasets,
        string? description = null)
    {
        if (string.IsNullOrWhiteSpace(snapshotName))
            throw new ArgumentException("Snapshot name cannot be null or empty.", nameof(snapshotName));

        if (datasets == null || datasets.Count == 0)
            throw new ArgumentException("At least one dataset must be included in the snapshot.", nameof(datasets));

        lock (SyncLock)
        {
            // Verify all datasets exist
            foreach (var kvp in datasets)
            {
                GetDatasetVersion(kvp.Key, kvp.Value);
            }

            var snapshot = new MultiDatasetSnapshot
            {
                SnapshotId = Guid.NewGuid().ToString("N"),
                SnapshotName = snapshotName,
                Datasets = new Dictionary<string, string>(datasets),
                Description = description,
                CreatedAt = DateTime.UtcNow
            };

            _snapshots[snapshotName] = snapshot;
            SaveSnapshots();

            return snapshot.SnapshotId;
        }
    }

    /// <summary>
    /// Retrieves a dataset snapshot.
    /// </summary>
    public override DatasetSnapshot GetDatasetSnapshot(string snapshotName)
    {
        if (string.IsNullOrWhiteSpace(snapshotName))
            throw new ArgumentException("Snapshot name cannot be null or empty.", nameof(snapshotName));

        lock (SyncLock)
        {
            if (!_snapshots.TryGetValue(snapshotName, out var multiSnapshot))
                throw new ArgumentException($"Snapshot '{snapshotName}' not found.", nameof(snapshotName));

            // Return as DatasetSnapshot (first dataset in the snapshot)
            var firstDataset = multiSnapshot.Datasets.First();
            var version = GetDatasetVersion(firstDataset.Key, firstDataset.Value);

            return new DatasetSnapshot
            {
                SnapshotId = multiSnapshot.SnapshotId,
                DatasetName = multiSnapshot.SnapshotName,
                Version = version.Version,
                SnapshotTime = multiSnapshot.CreatedAt,
                Description = multiSnapshot.Description,
                StoragePath = version.StoragePath,
                Hash = version.Hash
            };
        }
    }

    #region Private Helper Methods

    private string GetVersionFilePath(string datasetName, int version)
    {
        var datasetDir = GetDatasetDirectoryPath(datasetName);
        return Path.Combine(datasetDir, $"v{version}.json");
    }

    private long GetDataSize(string dataPath)
    {
        if (File.Exists(dataPath))
        {
            return new FileInfo(dataPath).Length;
        }
        else if (Directory.Exists(dataPath))
        {
            return Directory.GetFiles(dataPath, "*", SearchOption.AllDirectories)
                .Sum(f => new FileInfo(f).Length);
        }
        return 0;
    }

    private void LoadExistingData()
    {
        if (!Directory.Exists(StorageDirectory))
            return;

        // Load datasets
        foreach (var datasetDir in Directory.GetDirectories(StorageDirectory))
        {
            var datasetName = Path.GetFileName(datasetDir);
            var versionFiles = Directory.GetFiles(datasetDir, "v*.json");

            foreach (var versionFile in versionFiles)
            {
                try
                {
                    ValidatePathWithinDirectory(versionFile, StorageDirectory);
                    var json = File.ReadAllText(versionFile);
                    var version = DeserializeFromJson<DatasetVersion<T>>(json);

                    if (version != null)
                    {
                        if (!_datasets.ContainsKey(datasetName))
                        {
                            _datasets[datasetName] = new List<DatasetVersion<T>>();
                        }
                        _datasets[datasetName].Add(version);
                    }
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"[DataVersionControl] Failed to read version file '{versionFile}': {ex.Message}");
                }
                catch (JsonException ex)
                {
                    Console.WriteLine($"[DataVersionControl] Failed to deserialize version file '{versionFile}': {ex.Message}");
                }
            }
        }

        // Load metadata files
        LoadLineage();
        LoadVersionTags();
        LoadRunLinks();
        LoadSnapshots();
    }

    private void SaveDatasetVersion(DatasetVersion<T> version)
    {
        var datasetDir = GetDatasetDirectoryPath(version.DatasetName);
        if (!Directory.Exists(datasetDir))
        {
            Directory.CreateDirectory(datasetDir);
        }

        var filePath = GetVersionFilePath(version.DatasetName, version.Version);
        ValidatePathWithinDirectory(filePath, StorageDirectory);

        var json = SerializeToJson(version);
        File.WriteAllText(filePath, json);
    }

    private void LoadLineage()
    {
        var lineagePath = Path.Combine(StorageDirectory, "lineage.json");
        if (File.Exists(lineagePath))
        {
            try
            {
                var json = File.ReadAllText(lineagePath);
                var loaded = DeserializeFromJson<Dictionary<string, DatasetLineage>>(json);
                if (loaded != null)
                {
                    foreach (var kvp in loaded)
                    {
                        _lineage[kvp.Key] = kvp.Value;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DataVersionControl] Failed to load lineage: {ex.Message}");
            }
        }
    }

    private void SaveLineage()
    {
        var lineagePath = Path.Combine(StorageDirectory, "lineage.json");
        var json = SerializeToJson(_lineage);
        File.WriteAllText(lineagePath, json);
    }

    private void LoadVersionTags()
    {
        var tagsPath = Path.Combine(StorageDirectory, "tags.json");
        if (File.Exists(tagsPath))
        {
            try
            {
                var json = File.ReadAllText(tagsPath);
                var loaded = DeserializeFromJson<Dictionary<string, Dictionary<string, string>>>(json);
                if (loaded != null)
                {
                    foreach (var kvp in loaded)
                    {
                        _versionTags[kvp.Key] = kvp.Value;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DataVersionControl] Failed to load tags: {ex.Message}");
            }
        }
    }

    private void SaveVersionTags()
    {
        var tagsPath = Path.Combine(StorageDirectory, "tags.json");
        var json = SerializeToJson(_versionTags);
        File.WriteAllText(tagsPath, json);
    }

    private void LoadRunLinks()
    {
        var linksPath = Path.Combine(StorageDirectory, "run_links.json");
        if (File.Exists(linksPath))
        {
            try
            {
                var json = File.ReadAllText(linksPath);
                var loaded = DeserializeFromJson<Dictionary<string, RunLinkData>>(json);
                if (loaded != null)
                {
                    foreach (var kvp in loaded)
                    {
                        _runLinks[kvp.Key] = (kvp.Value.DatasetName, kvp.Value.VersionHash, kvp.Value.ModelId);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DataVersionControl] Failed to load run links: {ex.Message}");
            }
        }
    }

    private void SaveRunLinks()
    {
        var linksPath = Path.Combine(StorageDirectory, "run_links.json");
        var data = _runLinks.ToDictionary(
            kvp => kvp.Key,
            kvp => new RunLinkData
            {
                DatasetName = kvp.Value.DatasetName,
                VersionHash = kvp.Value.VersionHash,
                ModelId = kvp.Value.ModelId
            });
        var json = SerializeToJson(data);
        File.WriteAllText(linksPath, json);
    }

    private void LoadSnapshots()
    {
        var snapshotsPath = Path.Combine(StorageDirectory, "snapshots.json");
        if (File.Exists(snapshotsPath))
        {
            try
            {
                var json = File.ReadAllText(snapshotsPath);
                var loaded = DeserializeFromJson<Dictionary<string, MultiDatasetSnapshot>>(json);
                if (loaded != null)
                {
                    foreach (var kvp in loaded)
                    {
                        _snapshots[kvp.Key] = kvp.Value;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DataVersionControl] Failed to load snapshots: {ex.Message}");
            }
        }
    }

    private void SaveSnapshots()
    {
        var snapshotsPath = Path.Combine(StorageDirectory, "snapshots.json");
        var json = SerializeToJson(_snapshots);
        File.WriteAllText(snapshotsPath, json);
    }

    #endregion

    #region Nested Types

    private class RunLinkData
    {
        public string DatasetName { get; set; } = string.Empty;
        public string VersionHash { get; set; } = string.Empty;
        public string? ModelId { get; set; }
    }

    private class MultiDatasetSnapshot
    {
        public string SnapshotId { get; set; } = string.Empty;
        public string SnapshotName { get; set; } = string.Empty;
        public Dictionary<string, string> Datasets { get; set; } = new();
        public string? Description { get; set; }
        public DateTime CreatedAt { get; set; }
    }

    #endregion
}
