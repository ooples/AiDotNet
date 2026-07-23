using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;

namespace AiDotNet.DataVersioning;

/// <summary>
/// DVC-equivalent data version control system for ML datasets.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This class provides Git-like version control for your datasets.
/// It tracks changes, maintains history, and ensures reproducibility.
///
/// Key features:
/// - Content-addressable storage (files identified by hash)
/// - Efficient deduplication (same content stored once)
/// - Full version history with diff capabilities
/// - Data lineage tracking
/// - Metadata and tagging support
///
/// Example usage:
/// <code>
/// var dvc = new DataVersionControl("./data-versions");
///
/// // Create a dataset
/// var datasetId = dvc.CreateDataset("training-data", "MNIST training images");
///
/// // Add initial version
/// var v1 = dvc.AddVersion(datasetId, "./raw-data/mnist-train", "Initial import");
///
/// // After preprocessing, add new version
/// var v2 = dvc.AddVersion(datasetId, "./processed-data/mnist-train", "Normalized and augmented");
///
/// // Record the transformation
/// dvc.RecordLineage(datasetId, v2.VersionId,
///     new List&lt;(string, string)&gt; { (datasetId, v1.VersionId) },
///     "normalize_and_augment",
///     new Dictionary&lt;string, object&gt; { ["augmentation_factor"] = 5 });
///
/// // Get data path for training
/// var dataPath = dvc.GetDataPath(datasetId, "latest");
/// </code>
/// </remarks>
public class DataVersionControl : IDataVersionControl
{
    private readonly string _storageDirectory;
    private readonly string _datasetsDirectory;
    private readonly string _objectsDirectory;
    private readonly string _lineageDirectory;
    private readonly Dictionary<string, DatasetInfo> _datasets;
    private readonly Dictionary<string, List<DataVersion>> _versions;
    private readonly Dictionary<string, DataLineage> _lineage;
    private readonly object _lock = new();
    private bool _isDisposed;

    /// <inheritdoc/>
    public string StorageDirectory => _storageDirectory;

    /// <summary>
    /// Initializes a new instance of the DataVersionControl class.
    /// </summary>
    /// <param name="storageDirectory">Directory to store versioned data. Defaults to "./data-versions".</param>
    public DataVersionControl(string? storageDirectory = null)
    {
        _storageDirectory = storageDirectory ?? Path.Combine(Directory.GetCurrentDirectory(), "data-versions");
        _datasetsDirectory = Path.Combine(_storageDirectory, "datasets");
        _objectsDirectory = Path.Combine(_storageDirectory, "objects");
        _lineageDirectory = Path.Combine(_storageDirectory, "lineage");

        _datasets = new Dictionary<string, DatasetInfo>();
        _versions = new Dictionary<string, List<DataVersion>>();
        _lineage = new Dictionary<string, DataLineage>();

        EnsureDirectoriesExist();
        LoadExistingData();
    }

    /// <inheritdoc/>
    public string CreateDataset(string name, string? description = null, Dictionary<string, string>? metadata = null)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Dataset name cannot be empty.", nameof(name));

        lock (_lock)
        {
            // Check if dataset with same name exists
            var existing = _datasets.Values.FirstOrDefault(d => d.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (existing != null)
            {
                return existing.DatasetId;
            }

            var datasetId = GenerateId();
            var dataset = new DatasetInfo
            {
                DatasetId = datasetId,
                Name = name,
                Description = description,
                CreatedAt = DateTime.UtcNow,
                LastUpdatedAt = DateTime.UtcNow,
                VersionCount = 0,
                Metadata = metadata ?? new Dictionary<string, string>()
            };

            _datasets[datasetId] = dataset;
            _versions[datasetId] = new List<DataVersion>();

            // Create dataset directory
            var datasetDir = Path.Combine(_datasetsDirectory, datasetId);
            Directory.CreateDirectory(datasetDir);

            // Save metadata
            SaveDatasetMetadata(dataset);

            return datasetId;
        }
    }

    /// <inheritdoc/>
    public DataVersion AddVersion(string datasetId, string sourcePath, string? message = null, Dictionary<string, string>? metadata = null)
    {
        if (string.IsNullOrWhiteSpace(datasetId))
            throw new ArgumentException("Dataset ID cannot be empty.", nameof(datasetId));

        if (string.IsNullOrWhiteSpace(sourcePath))
            throw new ArgumentException("Source path cannot be empty.", nameof(sourcePath));

        if (!File.Exists(sourcePath) && !Directory.Exists(sourcePath))
            throw new FileNotFoundException($"Source path not found: {sourcePath}");

        lock (_lock)
        {
            if (!_datasets.TryGetValue(datasetId, out var dataset))
                throw new ArgumentException($"Dataset not found: {datasetId}", nameof(datasetId));

            // Calculate hashes and gather file info
            var files = GatherFileInfo(sourcePath);
            var contentHash = CalculateContentHash(files);
            var totalSize = files.Sum(f => f.SizeBytes);

            // Check if this exact content already exists
            var existingVersion = _versions[datasetId].FirstOrDefault(v => v.ContentHash == contentHash);
            if (existingVersion != null)
            {
                Console.WriteLine($"[DataVersionControl] Content unchanged - returning existing version {existingVersion.VersionId}");
                return existingVersion;
            }

            var versionNumber = _versions[datasetId].Count + 1;
            var versionId = contentHash.Substring(0, 12); // Use first 12 chars of hash

            var version = new DataVersion
            {
                VersionId = versionId,
                DatasetId = datasetId,
                VersionNumber = versionNumber,
                Message = message,
                CreatedAt = DateTime.UtcNow,
                ContentHash = contentHash,
                SizeBytes = totalSize,
                FileCount = files.Count,
                Files = files,
                Metadata = metadata ?? new Dictionary<string, string>()
            };

            // Store files in content-addressable storage
            var versionDir = Path.Combine(_datasetsDirectory, datasetId, versionId);
            Directory.CreateDirectory(versionDir);

            CopyToVersionStorage(sourcePath, versionDir, files);
            version.DataPath = versionDir;

            _versions[datasetId].Add(version);
            dataset.VersionCount = _versions[datasetId].Count;
            dataset.LatestVersionId = versionId;
            dataset.LastUpdatedAt = DateTime.UtcNow;

            // Save version metadata
            SaveVersionMetadata(version);
            SaveDatasetMetadata(dataset);

            return version;
        }
    }

    /// <inheritdoc/>
    public DataVersion GetVersion(string datasetId, string versionId = "latest")
    {
        lock (_lock)
        {
            if (!_versions.TryGetValue(datasetId, out var versions))
                throw new ArgumentException($"Dataset not found: {datasetId}", nameof(datasetId));

            if (versions.Count == 0)
                throw new InvalidOperationException($"Dataset {datasetId} has no versions.");

            if (versionId.Equals("latest", StringComparison.OrdinalIgnoreCase))
            {
                return versions.OrderByDescending(v => v.VersionNumber).First();
            }

            // Try exact match first
            var version = versions.FirstOrDefault(v => v.VersionId == versionId);
            if (version != null)
                return version;

            // Try version number
            if (int.TryParse(versionId, out var versionNum))
            {
                version = versions.FirstOrDefault(v => v.VersionNumber == versionNum);
                if (version != null)
                    return version;
            }

            throw new ArgumentException($"Version not found: {versionId}", nameof(versionId));
        }
    }

    /// <inheritdoc/>
    public List<DataVersion> ListVersions(string datasetId)
    {
        lock (_lock)
        {
            if (!_versions.TryGetValue(datasetId, out var versions))
                throw new ArgumentException($"Dataset not found: {datasetId}", nameof(datasetId));

            return versions.OrderByDescending(v => v.VersionNumber).ToList();
        }
    }

    /// <inheritdoc/>
    public List<DatasetInfo> ListDatasets()
    {
        lock (_lock)
        {
            return _datasets.Values.OrderByDescending(d => d.LastUpdatedAt).ToList();
        }
    }

    /// <inheritdoc/>
    public string GetDataPath(string datasetId, string versionId)
    {
        var version = GetVersion(datasetId, versionId);
        return version.DataPath ?? Path.Combine(_datasetsDirectory, datasetId, version.VersionId);
    }

    /// <inheritdoc/>
    public DataVersionDiff CompareVersions(string datasetId, string versionId1, string versionId2)
    {
        lock (_lock)
        {
            if (!_versions.TryGetValue(datasetId, out var versions))
                throw new ArgumentException($"Dataset not found: {datasetId}", nameof(datasetId));

            if (versions.Count == 0)
                throw new InvalidOperationException($"Dataset {datasetId} has no versions.");

            // Get both versions atomically within the same lock
            var v1 = GetVersionInternal(versions, versionId1, datasetId);
            var v2 = GetVersionInternal(versions, versionId2, datasetId);

            var v1Files = v1.Files.ToDictionary(f => f.RelativePath, f => f);
            var v2Files = v2.Files.ToDictionary(f => f.RelativePath, f => f);

            var diff = new DataVersionDiff
            {
                Version1 = v1,
                Version2 = v2
            };

            // Find added, removed, modified, unchanged
            foreach (var file in v2.Files)
            {
                if (!v1Files.TryGetValue(file.RelativePath, out var v1File))
                {
                    diff.FilesAdded.Add(file);
                }
                else if (v1File.Hash != file.Hash)
                {
                    diff.FilesModified.Add((v1File, file));
                }
                else
                {
                    diff.FilesUnchanged.Add(file);
                }
            }

            foreach (var file in v1.Files)
            {
                if (!v2Files.ContainsKey(file.RelativePath))
                {
                    diff.FilesRemoved.Add(file);
                }
            }

            return diff;
        }
    }

    /// <summary>
    /// Internal version lookup helper that operates within an existing lock.
    /// </summary>
    private static DataVersion GetVersionInternal(List<DataVersion> versions, string versionId, string datasetId)
    {
        if (versionId.Equals("latest", StringComparison.OrdinalIgnoreCase))
        {
            return versions.OrderByDescending(v => v.VersionNumber).First();
        }

        // Try exact match first
        var version = versions.FirstOrDefault(v => v.VersionId == versionId);
        if (version != null)
            return version;

        // Try version number
        if (int.TryParse(versionId, out var versionNum))
        {
            version = versions.FirstOrDefault(v => v.VersionNumber == versionNum);
            if (version != null)
                return version;
        }

        throw new ArgumentException($"Version not found: {versionId} in dataset {datasetId}");
    }

    /// <inheritdoc/>
    public void DeleteVersion(string datasetId, string versionId)
    {
        lock (_lock)
        {
            if (!_versions.TryGetValue(datasetId, out var versions))
                throw new ArgumentException($"Dataset not found: {datasetId}", nameof(datasetId));

            var version = versions.FirstOrDefault(v => v.VersionId == versionId);
            if (version == null)
                throw new ArgumentException($"Version not found: {versionId}", nameof(versionId));

            // Delete version directory
            var versionDir = Path.Combine(_datasetsDirectory, datasetId, versionId);
            if (Directory.Exists(versionDir))
            {
                Directory.Delete(versionDir, true);
            }

            // Delete version metadata
            var metaFile = Path.Combine(_datasetsDirectory, datasetId, $"{versionId}.json");
            if (File.Exists(metaFile))
            {
                File.Delete(metaFile);
            }

            versions.Remove(version);

            // Update dataset metadata
            if (_datasets.TryGetValue(datasetId, out var dataset))
            {
                dataset.VersionCount = versions.Count;
                dataset.LatestVersionId = versions.OrderByDescending(v => v.VersionNumber).FirstOrDefault()?.VersionId;
                dataset.LastUpdatedAt = DateTime.UtcNow;
                SaveDatasetMetadata(dataset);
            }
        }
    }

    /// <inheritdoc/>
    public void DeleteDataset(string datasetId)
    {
        lock (_lock)
        {
            if (!_datasets.ContainsKey(datasetId))
                throw new ArgumentException($"Dataset not found: {datasetId}", nameof(datasetId));

            // Delete dataset directory
            var datasetDir = Path.Combine(_datasetsDirectory, datasetId);
            if (Directory.Exists(datasetDir))
            {
                Directory.Delete(datasetDir, true);
            }

            // Delete lineage records from memory AND disk
            var lineageKeys = _lineage.Keys
                .Where(k => k.StartsWith($"{datasetId}/"))
                .ToList();
            foreach (var key in lineageKeys)
            {
                // Delete the lineage file from disk
                var parts = key.Split('/');
                if (parts.Length >= 2)
                {
                    var lineageFile = Path.Combine(_lineageDirectory, $"{parts[0]}_{parts[1]}.json");
                    if (File.Exists(lineageFile))
                    {
                        File.Delete(lineageFile);
                    }
                }

                _lineage.Remove(key);
            }

            _datasets.Remove(datasetId);
            _versions.Remove(datasetId);
        }
    }

    /// <inheritdoc/>
    public void RecordLineage(
        string outputDatasetId,
        string outputVersionId,
        List<(string datasetId, string versionId)> inputDatasets,
        string transformation,
        Dictionary<string, object>? parameters = null)
    {
        lock (_lock)
        {
            var lineage = new DataLineage
            {
                DatasetId = outputDatasetId,
                VersionId = outputVersionId,
                Inputs = inputDatasets,
                Transformation = transformation,
                Parameters = parameters,
                RecordedAt = DateTime.UtcNow
            };

            var key = $"{outputDatasetId}/{outputVersionId}";
            _lineage[key] = lineage;

            // Save to disk
            var lineageFile = Path.Combine(_lineageDirectory, $"{outputDatasetId}_{outputVersionId}.json");
            var json = JsonConvert.SerializeObject(lineage, Formatting.Indented);
            File.WriteAllText(lineageFile, json);
        }
    }

    /// <inheritdoc/>
    public DataLineage GetLineage(string datasetId, string versionId)
    {
        lock (_lock)
        {
            return GetLineageInternal(datasetId, versionId, new HashSet<string>());
        }
    }

    /// <summary>
    /// Internal helper for GetLineage that tracks visited nodes to prevent infinite recursion.
    /// Creates a copy of the cached lineage to avoid mutating in-memory state.
    /// </summary>
    private DataLineage GetLineageInternal(string datasetId, string versionId, HashSet<string> visited)
    {
        var key = $"{datasetId}/{versionId}";

        // Check for cycles to prevent infinite recursion
        if (!visited.Add(key))
        {
            return new DataLineage
            {
                DatasetId = datasetId,
                VersionId = versionId,
                UpstreamLineage = new List<DataLineage>()
            };
        }

        if (!_lineage.TryGetValue(key, out var cachedLineage))
        {
            // No lineage recorded - return empty lineage
            return new DataLineage
            {
                DatasetId = datasetId,
                VersionId = versionId,
                UpstreamLineage = new List<DataLineage>()
            };
        }

        // Create a copy to avoid mutating cached data
        var lineageCopy = new DataLineage
        {
            DatasetId = cachedLineage.DatasetId,
            VersionId = cachedLineage.VersionId,
            Inputs = cachedLineage.Inputs,
            Transformation = cachedLineage.Transformation,
            Parameters = cachedLineage.Parameters,
            RecordedAt = cachedLineage.RecordedAt,
            UpstreamLineage = new List<DataLineage>()
        };

        // Recursively get upstream lineage
        foreach (var (inputDatasetId, inputVersionId) in cachedLineage.Inputs)
        {
            var upstreamLineage = GetLineageInternal(inputDatasetId, inputVersionId, visited);
            lineageCopy.UpstreamLineage.Add(upstreamLineage);
        }

        return lineageCopy;
    }

    #region Private Helper Methods

    private void EnsureDirectoriesExist()
    {
        Directory.CreateDirectory(_storageDirectory);
        Directory.CreateDirectory(_datasetsDirectory);
        Directory.CreateDirectory(_objectsDirectory);
        Directory.CreateDirectory(_lineageDirectory);
    }

    private void LoadExistingData()
    {
        // Load datasets
        if (!Directory.Exists(_datasetsDirectory))
            return;

        foreach (var datasetDir in Directory.GetDirectories(_datasetsDirectory))
        {
            var metaFile = Path.Combine(datasetDir, "dataset.json");
            if (!File.Exists(metaFile))
                continue;

            try
            {
                var json = File.ReadAllText(metaFile);
                var dataset = JsonConvert.DeserializeObject<DatasetInfo>(json);
                if (dataset != null)
                {
                    _datasets[dataset.DatasetId] = dataset;
                    _versions[dataset.DatasetId] = new List<DataVersion>();

                    // Load versions
                    foreach (var versionFile in Directory.GetFiles(datasetDir, "*.json").Where(f => !f.EndsWith("dataset.json")))
                    {
                        try
                        {
                            var versionJson = File.ReadAllText(versionFile);
                            var version = JsonConvert.DeserializeObject<DataVersion>(versionJson);
                            if (version != null)
                            {
                                _versions[dataset.DatasetId].Add(version);
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[DataVersionControl] Skipped invalid version file '{versionFile}': {ex.Message}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DataVersionControl] Skipped invalid dataset directory '{datasetDir}': {ex.Message}");
            }
        }

        // Load lineage records
        if (Directory.Exists(_lineageDirectory))
        {
            foreach (var lineageFile in Directory.GetFiles(_lineageDirectory, "*.json"))
            {
                try
                {
                    var json = File.ReadAllText(lineageFile);
                    var lineage = JsonConvert.DeserializeObject<DataLineage>(json);
                    if (lineage != null)
                    {
                        var key = $"{lineage.DatasetId}/{lineage.VersionId}";
                        _lineage[key] = lineage;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[DataVersionControl] Skipped invalid lineage file '{lineageFile}': {ex.Message}");
                }
            }
        }
    }

    private static List<DataFileInfo> GatherFileInfo(string sourcePath)
    {
        var files = new List<DataFileInfo>();

        if (File.Exists(sourcePath))
        {
            var fileInfo = new FileInfo(sourcePath);
            files.Add(new DataFileInfo
            {
                RelativePath = fileInfo.Name,
                SizeBytes = fileInfo.Length,
                Hash = CalculateFileHash(sourcePath),
                LastModified = fileInfo.LastWriteTimeUtc
            });
        }
        else if (Directory.Exists(sourcePath))
        {
            var basePath = new Uri(sourcePath.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar);

            foreach (var filePath in Directory.GetFiles(sourcePath, "*", SearchOption.AllDirectories))
            {
                var fileUri = new Uri(filePath);
                var relativePath = Uri.UnescapeDataString(basePath.MakeRelativeUri(fileUri).ToString())
                    .Replace('/', Path.DirectorySeparatorChar);

                var fileInfo = new FileInfo(filePath);
                files.Add(new DataFileInfo
                {
                    RelativePath = relativePath,
                    SizeBytes = fileInfo.Length,
                    Hash = CalculateFileHash(filePath),
                    LastModified = fileInfo.LastWriteTimeUtc
                });
            }
        }

        return files.OrderBy(f => f.RelativePath).ToList();
    }

    private static string CalculateFileHash(string filePath)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);
        var hashBytes = sha256.ComputeHash(stream);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }

    private static string CalculateContentHash(List<DataFileInfo> files)
    {
        using var sha256 = SHA256.Create();
        var combinedHashes = string.Join(":", files.Select(f => $"{f.RelativePath}:{f.Hash}"));
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(combinedHashes));
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }

    private static void CopyToVersionStorage(string sourcePath, string versionDir, List<DataFileInfo> files)
    {
        if (File.Exists(sourcePath))
        {
            var destPath = Path.Combine(versionDir, Path.GetFileName(sourcePath));
            File.Copy(sourcePath, destPath, true);
        }
        else if (Directory.Exists(sourcePath))
        {
            foreach (var file in files)
            {
                var sourceFile = Path.Combine(sourcePath, file.RelativePath);
                var destFile = Path.Combine(versionDir, file.RelativePath);

                var destDir = Path.GetDirectoryName(destFile);
                if (destDir != null && !Directory.Exists(destDir))
                {
                    Directory.CreateDirectory(destDir);
                }

                File.Copy(sourceFile, destFile, true);
            }
        }
    }

    private void SaveDatasetMetadata(DatasetInfo dataset)
    {
        var datasetDir = Path.Combine(_datasetsDirectory, dataset.DatasetId);
        Directory.CreateDirectory(datasetDir);

        var metaFile = Path.Combine(datasetDir, "dataset.json");
        var json = JsonConvert.SerializeObject(dataset, Formatting.Indented);
        File.WriteAllText(metaFile, json);
    }

    private void SaveVersionMetadata(DataVersion version)
    {
        var metaFile = Path.Combine(_datasetsDirectory, version.DatasetId, $"{version.VersionId}.json");
        var json = JsonConvert.SerializeObject(version, Formatting.Indented);
        File.WriteAllText(metaFile, json);
    }

    private static string GenerateId()
    {
        return Guid.NewGuid().ToString("N").Substring(0, 12);
    }

    #endregion

    /// <summary>
    /// Disposes the data version control system.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;
    }
}
