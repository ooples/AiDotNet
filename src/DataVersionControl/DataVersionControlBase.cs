#if !NET6_0_OR_GREATER
#pragma warning disable CS8600, CS8601, CS8602, CS8603, CS8604
using AiDotNet.TrainingMonitoring;
#endif
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Newtonsoft.Json;
using System.Security.Cryptography;

namespace AiDotNet.DataVersionControl;

/// <summary>
/// Base class for data version control implementations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This abstract base class provides common functionality for data
/// version control systems. It handles storage path management, hash computation for
/// integrity verification, and data lineage tracking.
///
/// Key features:
/// - Path security validation
/// - SHA-256 hash computation for data integrity
/// - Dataset versioning support
/// - Lineage tracking for reproducibility
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public abstract class DataVersionControlBase<T> : IDataVersionControl<T>
{
    /// <summary>
    /// The directory where version control data is stored.
    /// </summary>
    protected readonly string StorageDirectory;

    /// <summary>
    /// Lock object for thread-safe operations.
    /// </summary>
    protected readonly object SyncLock = new();

    /// <summary>
    /// JSON serialization settings for consistent serialization.
    /// Uses TypeNameHandling.None for security - no type metadata in JSON output.
    /// </summary>
    protected static readonly JsonSerializerSettings JsonSettings = new()
    {
        Formatting = Formatting.Indented,
        TypeNameHandling = TypeNameHandling.None
    };

    /// <summary>
    /// Initializes a new instance of the DataVersionControlBase class.
    /// </summary>
    /// <param name="storageDirectory">Directory to store version control data.</param>
    /// <param name="baseDirectory">Base directory for path validation. Defaults to current directory.</param>
    /// <remarks>
    /// When a custom storageDirectory is provided without a baseDirectory, the storage directory
    /// itself becomes the base for path validation. This allows users to store version control data in
    /// any location they choose (like temp directories for tests) while still preventing
    /// path traversal attacks within that chosen directory.
    /// </remarks>
    protected DataVersionControlBase(string? storageDirectory = null, string? baseDirectory = null)
    {
        if (storageDirectory != null)
        {
            // User provided a custom storage directory
            // Use the provided base directory, or default to the storage directory itself
            var fullStoragePath = Path.GetFullPath(storageDirectory);
            var baseDir = baseDirectory != null ? Path.GetFullPath(baseDirectory) : fullStoragePath;
            StorageDirectory = GetSanitizedPath(fullStoragePath, baseDir);
        }
        else
        {
            // Using default storage directory - validate against base
            var baseDir = baseDirectory ?? Directory.GetCurrentDirectory();
            var defaultStorage = Path.Combine(baseDir, "data_versions");
            StorageDirectory = GetSanitizedPath(defaultStorage, baseDir);
        }

        EnsureStorageDirectoryExists();
    }

    /// <summary>
    /// Creates a new dataset version.
    /// </summary>
    public abstract string CreateDatasetVersion(
        string datasetName,
        string dataPath,
        string? description = null,
        Dictionary<string, object>? metadata = null,
        Dictionary<string, string>? tags = null);

    /// <summary>
    /// Retrieves a specific version of a dataset.
    /// </summary>
    public abstract DatasetVersion<T> GetDatasetVersion(string datasetName, string? versionHash = null);

    /// <summary>
    /// Gets the latest version of a dataset.
    /// </summary>
    public abstract DatasetVersion<T> GetLatestDatasetVersion(string datasetName);

    /// <summary>
    /// Lists all versions of a dataset.
    /// </summary>
    public abstract List<DatasetVersionInfo<T>> ListDatasetVersions(string datasetName);

    /// <summary>
    /// Lists all tracked datasets.
    /// </summary>
    public abstract List<string> ListDatasets(string? filter = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Computes and stores a hash of the dataset for integrity verification.
    /// </summary>
    /// <remarks>
    /// Uses streaming/incremental hashing for memory efficiency when processing
    /// directories with many or large files. Files are processed one at a time
    /// rather than loading all content into memory.
    /// </remarks>
    public virtual string ComputeDatasetHash(string dataPath)
    {
        if (string.IsNullOrWhiteSpace(dataPath))
            throw new ArgumentException("Data path cannot be null or empty.", nameof(dataPath));

        var fullPath = Path.GetFullPath(dataPath);
        if (!File.Exists(fullPath) && !Directory.Exists(fullPath))
            throw new FileNotFoundException($"Data path not found: {fullPath}");

        using var sha256 = SHA256.Create();

        if (File.Exists(fullPath))
        {
            using var stream = File.OpenRead(fullPath);
            var hash = sha256.ComputeHash(stream);
            return Convert.ToBase64String(hash);
        }
        else
        {
            // For directories, use incremental hashing to avoid memory issues
            // with large directories containing many files
            var files = Directory.GetFiles(fullPath, "*", SearchOption.AllDirectories)
                .OrderBy(f => f, StringComparer.Ordinal)
                .ToList();

            // Use a buffer for streaming file content through the hash
            var buffer = new byte[81920]; // 80KB buffer - good balance for I/O

            foreach (var file in files)
            {
#if NET6_0_OR_GREATER
                var relativePath = Path.GetRelativePath(fullPath, file);
#else
                var relativePath = FrameworkPolyfills.GetRelativePath(fullPath, file);
#endif
                // Hash the relative path (includes path in hash for integrity)
                var pathBytes = System.Text.Encoding.UTF8.GetBytes(relativePath);
                sha256.TransformBlock(pathBytes, 0, pathBytes.Length, pathBytes, 0);

                // Stream file content through hash without loading entire file
                using var fileStream = File.OpenRead(file);
                int bytesRead;
                while ((bytesRead = fileStream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    sha256.TransformBlock(buffer, 0, bytesRead, buffer, 0);
                }
            }

            // Finalize the hash
            sha256.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
            return Convert.ToBase64String(sha256.Hash ?? Array.Empty<byte>());
        }
    }

    /// <summary>
    /// Verifies that a dataset hasn't been modified by comparing its hash.
    /// </summary>
    public virtual bool VerifyDatasetIntegrity(string datasetName, string versionHash, string currentDataPath)
    {
        var currentHash = ComputeDatasetHash(currentDataPath);
        return string.Equals(currentHash, versionHash, StringComparison.Ordinal);
    }

    /// <summary>
    /// Links a dataset version to a model training run.
    /// </summary>
    public abstract void LinkDatasetToRun(string datasetName, string versionHash, string runId, string? modelId = null);

    /// <summary>
    /// Gets all training runs that used a specific dataset version.
    /// </summary>
    public abstract List<string> GetRunsUsingDataset(string datasetName, string versionHash);

    /// <summary>
    /// Gets the dataset version used by a specific training run.
    /// </summary>
    public abstract DatasetVersion<T> GetDatasetForRun(string runId);

    /// <summary>
    /// Tags a dataset version for easy reference.
    /// </summary>
    public abstract void TagDatasetVersion(string datasetName, string versionHash, string tag);

    /// <summary>
    /// Gets a dataset version by its tag.
    /// </summary>
    public abstract DatasetVersion<T> GetDatasetByTag(string datasetName, string tag);

    /// <summary>
    /// Compares two dataset versions to see what changed.
    /// </summary>
    public abstract DatasetComparison<T> CompareDatasetVersions(string datasetName, string version1Hash, string version2Hash);

    /// <summary>
    /// Records metadata about how a dataset was created or transformed.
    /// </summary>
    public abstract void RecordDatasetLineage(string datasetName, string versionHash, DatasetLineage lineage);

    /// <summary>
    /// Gets the lineage information for a dataset version.
    /// </summary>
    public abstract DatasetLineage GetDatasetLineage(string datasetName, string versionHash);

    /// <summary>
    /// Deletes a specific dataset version.
    /// </summary>
    public abstract void DeleteDatasetVersion(string datasetName, string versionHash);

    /// <summary>
    /// Gets statistics about a dataset version.
    /// </summary>
    public abstract DatasetStatistics<T> GetDatasetStatistics(string datasetName, string versionHash);

    /// <summary>
    /// Creates a snapshot of multiple related datasets together.
    /// </summary>
    public abstract string CreateDatasetSnapshot(
        string snapshotName,
        Dictionary<string, string> datasets,
        string? description = null);

    /// <summary>
    /// Retrieves a dataset snapshot, returning only the first dataset in the snapshot.
    /// </summary>
    /// <remarks>
    /// <b>Important:</b> This method is designed for single-dataset snapshots or when you only
    /// need information about one dataset. For multi-dataset snapshots created with
    /// <see cref="CreateDatasetSnapshot"/>, this method returns only the first dataset's information.
    /// Use <see cref="GetAllDatasetsInSnapshot"/> to retrieve all datasets in a multi-dataset snapshot.
    /// </remarks>
    /// <param name="snapshotName">The name of the snapshot to retrieve.</param>
    /// <returns>A DatasetSnapshot containing information about the first dataset in the snapshot.</returns>
    public abstract DatasetSnapshot GetDatasetSnapshot(string snapshotName);

    /// <summary>
    /// Retrieves all datasets in a multi-dataset snapshot.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When you create a snapshot with multiple datasets using
    /// <see cref="CreateDatasetSnapshot"/>, use this method to retrieve all of them.
    /// Each entry in the returned dictionary maps a dataset name to its version hash.
    /// </remarks>
    /// <param name="snapshotName">The name of the snapshot to retrieve.</param>
    /// <returns>
    /// A tuple containing:
    /// - SnapshotId: The unique identifier of the snapshot
    /// - Datasets: A dictionary mapping dataset names to their version hashes
    /// - Description: The optional description of the snapshot
    /// - CreatedAt: When the snapshot was created
    /// </returns>
    public abstract (string SnapshotId, Dictionary<string, string> Datasets, string? Description, DateTime CreatedAt) GetAllDatasetsInSnapshot(string snapshotName);

    #region Protected Helper Methods

    /// <summary>
    /// Ensures the storage directory exists.
    /// </summary>
    protected virtual void EnsureStorageDirectoryExists()
    {
        if (!Directory.Exists(StorageDirectory))
        {
            Directory.CreateDirectory(StorageDirectory);
        }
    }

    /// <summary>
    /// Gets the directory path for a dataset.
    /// </summary>
    /// <param name="datasetName">The dataset name.</param>
    /// <returns>The sanitized directory path for the dataset.</returns>
    protected virtual string GetDatasetDirectoryPath(string datasetName)
    {
        var sanitizedName = GetSanitizedFileName(datasetName);
        var path = Path.Combine(StorageDirectory, sanitizedName);
        ValidatePathWithinDirectory(path, StorageDirectory);
        return path;
    }

    /// <summary>
    /// Validates that a dataset name is valid.
    /// </summary>
    /// <param name="datasetName">The dataset name to validate.</param>
    /// <exception cref="ArgumentException">Thrown when the dataset name is invalid.</exception>
    protected virtual void ValidateDatasetName(string datasetName)
    {
        if (string.IsNullOrWhiteSpace(datasetName))
            throw new ArgumentException("Dataset name cannot be null or empty.", nameof(datasetName));
    }

    /// <summary>
    /// Serializes an object to JSON.
    /// </summary>
    protected virtual string SerializeToJson(object obj)
    {
        return JsonConvert.SerializeObject(obj, JsonSettings);
    }

    /// <summary>
    /// Deserializes a JSON string to an object.
    /// </summary>
    protected virtual TResult? DeserializeFromJson<TResult>(string json) where TResult : class
    {
        return JsonConvert.DeserializeObject<TResult>(json, JsonSettings);
    }

    /// <summary>
    /// Sanitizes a file name to prevent path traversal attacks.
    /// </summary>
    protected static string GetSanitizedFileName(string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            throw new ArgumentException("File name cannot be null or empty.", nameof(fileName));

        var sanitized = Path.GetFileName(fileName);
        if (string.IsNullOrWhiteSpace(sanitized))
            throw new ArgumentException("Invalid file name after sanitization.", nameof(fileName));

        return sanitized;
    }

    /// <summary>
    /// Gets a sanitized path, ensuring it doesn't escape the base directory.
    /// </summary>
    /// <remarks>
    /// This method enforces strict path containment. All paths must resolve to within
    /// the base directory or match it exactly. Rooted paths that point outside the
    /// base directory are rejected to prevent directory traversal attacks.
    /// </remarks>
    protected static string GetSanitizedPath(string path, string baseDirectory)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));

        var fullPath = Path.GetFullPath(path);
        var fullBaseDir = Path.GetFullPath(baseDirectory);

        // Normalize paths for comparison (ensure they end consistently)
        if (!fullBaseDir.EndsWith(Path.DirectorySeparatorChar.ToString()))
        {
            fullBaseDir += Path.DirectorySeparatorChar;
        }

        // Check if path is within base directory or equals it
        bool isWithinBase = fullPath.StartsWith(fullBaseDir, StringComparison.OrdinalIgnoreCase) ||
                            fullPath.Equals(fullBaseDir.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase);

        if (!isWithinBase)
        {
            throw new ArgumentException($"Path '{path}' is outside the allowed directory '{baseDirectory}'.", nameof(path));
        }

        return fullPath;
    }

    /// <summary>
    /// Validates that a path is within the specified directory.
    /// </summary>
    protected static void ValidatePathWithinDirectory(string path, string directory)
    {
        var fullPath = Path.GetFullPath(path);
        var fullDir = Path.GetFullPath(directory);

        // Normalize directory path with trailing separator to prevent sibling-prefix bypasses
        // (e.g., "C:\app\datamalicious" should not match "C:\app\data")
        if (!fullDir.EndsWith(Path.DirectorySeparatorChar.ToString()))
        {
            fullDir += Path.DirectorySeparatorChar;
        }

        if (!fullPath.StartsWith(fullDir, StringComparison.OrdinalIgnoreCase))
        {
            throw new UnauthorizedAccessException($"Access to path '{path}' is denied. Path must be within '{directory}'.");
        }
    }

    #endregion
}
