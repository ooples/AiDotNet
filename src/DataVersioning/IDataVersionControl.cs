namespace AiDotNet.DataVersioning;

/// <summary>
/// Interface for data version control systems.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Data version control (like DVC) helps you track
/// changes to your datasets, ensuring you can always reproduce experiments
/// with the exact same data.
///
/// Key concepts:
/// - Dataset: A collection of data files (training data, validation data, etc.)
/// - Version: A specific snapshot of a dataset at a point in time
/// - Hash: A unique identifier based on file contents
/// - Lineage: The history of how data was transformed
/// </remarks>
public interface IDataVersionControl : IDisposable
{
    /// <summary>
    /// Gets the storage directory for data versions.
    /// </summary>
    string StorageDirectory { get; }

    /// <summary>
    /// Registers a new dataset for version control.
    /// </summary>
    /// <param name="name">The dataset name.</param>
    /// <param name="description">Optional description.</param>
    /// <param name="metadata">Optional metadata.</param>
    /// <returns>The dataset ID.</returns>
    string CreateDataset(string name, string? description = null, Dictionary<string, string>? metadata = null);

    /// <summary>
    /// Adds a new version of a dataset.
    /// </summary>
    /// <param name="datasetId">The dataset ID.</param>
    /// <param name="sourcePath">Path to the data (file or directory).</param>
    /// <param name="message">Version message describing the changes.</param>
    /// <param name="metadata">Optional version metadata.</param>
    /// <returns>The version information.</returns>
    DataVersion AddVersion(string datasetId, string sourcePath, string? message = null, Dictionary<string, string>? metadata = null);

    /// <summary>
    /// Gets a specific version of a dataset.
    /// </summary>
    /// <param name="datasetId">The dataset ID.</param>
    /// <param name="versionId">The version ID (or "latest" for most recent).</param>
    /// <returns>The version information.</returns>
    DataVersion GetVersion(string datasetId, string versionId = "latest");

    /// <summary>
    /// Lists all versions of a dataset.
    /// </summary>
    /// <param name="datasetId">The dataset ID.</param>
    /// <returns>List of versions ordered by date descending.</returns>
    List<DataVersion> ListVersions(string datasetId);

    /// <summary>
    /// Lists all registered datasets.
    /// </summary>
    /// <returns>List of datasets.</returns>
    List<DatasetInfo> ListDatasets();

    /// <summary>
    /// Gets the path to access a specific version's data.
    /// </summary>
    /// <param name="datasetId">The dataset ID.</param>
    /// <param name="versionId">The version ID.</param>
    /// <returns>Path to the versioned data.</returns>
    string GetDataPath(string datasetId, string versionId);

    /// <summary>
    /// Compares two versions of a dataset.
    /// </summary>
    /// <param name="datasetId">The dataset ID.</param>
    /// <param name="versionId1">First version ID.</param>
    /// <param name="versionId2">Second version ID.</param>
    /// <returns>Comparison results.</returns>
    DataVersionDiff CompareVersions(string datasetId, string versionId1, string versionId2);

    /// <summary>
    /// Deletes a specific version.
    /// </summary>
    /// <param name="datasetId">The dataset ID.</param>
    /// <param name="versionId">The version ID.</param>
    void DeleteVersion(string datasetId, string versionId);

    /// <summary>
    /// Deletes a dataset and all its versions.
    /// </summary>
    /// <param name="datasetId">The dataset ID.</param>
    void DeleteDataset(string datasetId);

    /// <summary>
    /// Records data lineage (transformation history).
    /// </summary>
    /// <param name="outputDatasetId">The output dataset ID.</param>
    /// <param name="outputVersionId">The output version ID.</param>
    /// <param name="inputDatasets">Input datasets and versions.</param>
    /// <param name="transformation">Description of the transformation.</param>
    /// <param name="parameters">Transformation parameters.</param>
    void RecordLineage(
        string outputDatasetId,
        string outputVersionId,
        List<(string datasetId, string versionId)> inputDatasets,
        string transformation,
        Dictionary<string, object>? parameters = null);

    /// <summary>
    /// Gets the lineage (data ancestry) for a version.
    /// </summary>
    /// <param name="datasetId">The dataset ID.</param>
    /// <param name="versionId">The version ID.</param>
    /// <returns>Lineage information.</returns>
    DataLineage GetLineage(string datasetId, string versionId);
}

/// <summary>
/// Represents a dataset registered for version control.
/// </summary>
public class DatasetInfo
{
    /// <summary>
    /// Gets or sets the unique dataset identifier.
    /// </summary>
    public string DatasetId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the dataset name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the dataset description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets when the dataset was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets when the dataset was last updated.
    /// </summary>
    public DateTime LastUpdatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the number of versions.
    /// </summary>
    public int VersionCount { get; set; }

    /// <summary>
    /// Gets or sets the latest version ID.
    /// </summary>
    public string? LatestVersionId { get; set; }

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Represents a specific version of a dataset.
/// </summary>
public class DataVersion
{
    /// <summary>
    /// Gets or sets the version identifier (content hash).
    /// </summary>
    public string VersionId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the dataset this version belongs to.
    /// </summary>
    public string DatasetId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version number (sequential).
    /// </summary>
    public int VersionNumber { get; set; }

    /// <summary>
    /// Gets or sets the version message.
    /// </summary>
    public string? Message { get; set; }

    /// <summary>
    /// Gets or sets when this version was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the content hash (SHA-256).
    /// </summary>
    public string ContentHash { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the total size in bytes.
    /// </summary>
    public long SizeBytes { get; set; }

    /// <summary>
    /// Gets or sets the number of files.
    /// </summary>
    public int FileCount { get; set; }

    /// <summary>
    /// Gets or sets file-level information.
    /// </summary>
    public List<DataFileInfo> Files { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets the path to the versioned data.
    /// </summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Gets the size in a human-readable format.
    /// </summary>
    public string SizeFormatted
    {
        get
        {
            if (SizeBytes < 1024) return $"{SizeBytes} B";
            if (SizeBytes < 1024 * 1024) return $"{SizeBytes / 1024.0:F1} KB";
            if (SizeBytes < 1024 * 1024 * 1024) return $"{SizeBytes / (1024.0 * 1024):F1} MB";
            return $"{SizeBytes / (1024.0 * 1024 * 1024):F2} GB";
        }
    }
}

/// <summary>
/// Information about a single file in a data version.
/// </summary>
public class DataFileInfo
{
    /// <summary>
    /// Gets or sets the relative file path.
    /// </summary>
    public string RelativePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the file size in bytes.
    /// </summary>
    public long SizeBytes { get; set; }

    /// <summary>
    /// Gets or sets the file content hash.
    /// </summary>
    public string Hash { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the last modified time.
    /// </summary>
    public DateTime LastModified { get; set; }
}

/// <summary>
/// Represents differences between two data versions.
/// </summary>
public class DataVersionDiff
{
    /// <summary>
    /// Gets or sets the first version.
    /// </summary>
    public DataVersion Version1 { get; set; } = new();

    /// <summary>
    /// Gets or sets the second version.
    /// </summary>
    public DataVersion Version2 { get; set; } = new();

    /// <summary>
    /// Gets or sets files added in version 2.
    /// </summary>
    public List<DataFileInfo> FilesAdded { get; set; } = new();

    /// <summary>
    /// Gets or sets files removed in version 2.
    /// </summary>
    public List<DataFileInfo> FilesRemoved { get; set; } = new();

    /// <summary>
    /// Gets or sets files modified between versions.
    /// </summary>
    public List<(DataFileInfo before, DataFileInfo after)> FilesModified { get; set; } = new();

    /// <summary>
    /// Gets or sets unchanged files.
    /// </summary>
    public List<DataFileInfo> FilesUnchanged { get; set; } = new();

    /// <summary>
    /// Gets the size change in bytes.
    /// </summary>
    public long SizeDelta => Version2.SizeBytes - Version1.SizeBytes;

    /// <summary>
    /// Gets a summary of changes.
    /// </summary>
    public string Summary =>
        $"Added: {FilesAdded.Count}, Removed: {FilesRemoved.Count}, Modified: {FilesModified.Count}, Unchanged: {FilesUnchanged.Count}";
}

/// <summary>
/// Data lineage information showing data ancestry and transformations.
/// </summary>
public class DataLineage
{
    /// <summary>
    /// Gets or sets the dataset ID.
    /// </summary>
    public string DatasetId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version ID.
    /// </summary>
    public string VersionId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the input datasets and versions.
    /// </summary>
    public List<(string datasetId, string versionId)> Inputs { get; set; } = new();

    /// <summary>
    /// Gets or sets the transformation description.
    /// </summary>
    public string? Transformation { get; set; }

    /// <summary>
    /// Gets or sets transformation parameters.
    /// </summary>
    public Dictionary<string, object>? Parameters { get; set; }

    /// <summary>
    /// Gets or sets when the transformation was recorded.
    /// </summary>
    public DateTime RecordedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets upstream lineage (recursive).
    /// </summary>
    public List<DataLineage> UpstreamLineage { get; set; } = new();
}
