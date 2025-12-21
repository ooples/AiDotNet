namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for data version control systems that track dataset changes over time.
/// </summary>
/// <remarks>
/// A data version control system manages versions of datasets used for training and evaluating models,
/// ensuring reproducibility and traceability.
///
/// <b>For Beginners:</b> Think of data version control like Git, but for your datasets instead of code.
/// Just like Git tracks changes to your code, data version control tracks changes to your data:
/// - Records what data was used to train each model
/// - Lets you go back to previous versions of datasets
/// - Helps reproduce experiments with exact same data
/// - Tracks where data came from and how it was transformed
///
/// Common scenarios include:
/// - Dataset updates (new examples added, errors corrected)
/// - Data preprocessing changes (different normalization, feature engineering)
/// - Train/validation/test splits that need to be reproduced
/// - Tracking data lineage for compliance
///
/// Why data version control matters:
/// - Models trained on different data versions perform differently
/// - Reproducing results requires exact same data
/// - Debugging requires knowing what data was used
/// - Compliance and auditing need data traceability
/// - Collaboration requires shared understanding of data versions
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IDataVersionControl<T>
{
    /// <summary>
    /// Creates a new dataset version.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This saves a snapshot of your dataset with a unique identifier,
    /// like committing changes in Git.
    /// </remarks>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="dataPath">Path to the data file(s).</param>
    /// <param name="description">Description of this version.</param>
    /// <param name="metadata">Additional metadata about the dataset.</param>
    /// <param name="tags">Tags for categorizing the dataset.</param>
    /// <returns>The unique identifier (version hash) for this dataset version.</returns>
    string CreateDatasetVersion(
        string datasetName,
        string dataPath,
        string? description = null,
        Dictionary<string, object>? metadata = null,
        Dictionary<string, string>? tags = null);

    /// <summary>
    /// Retrieves a specific version of a dataset.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">The version hash to retrieve. If null, gets latest.</param>
    /// <returns>Information about the dataset version.</returns>
    DatasetVersion<T> GetDatasetVersion(string datasetName, string? versionHash = null);

    /// <summary>
    /// Gets the latest version of a dataset.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <returns>The latest version of the dataset.</returns>
    DatasetVersion<T> GetLatestDatasetVersion(string datasetName);

    /// <summary>
    /// Lists all versions of a dataset.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <returns>List of all dataset versions with metadata.</returns>
    List<DatasetVersionInfo<T>> ListDatasetVersions(string datasetName);

    /// <summary>
    /// Lists all tracked datasets.
    /// </summary>
    /// <param name="filter">Optional filter expression.</param>
    /// <param name="tags">Optional tags to filter by.</param>
    /// <returns>List of dataset names matching the criteria.</returns>
    List<string> ListDatasets(string? filter = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Computes and stores a hash of the dataset for integrity verification.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A hash is like a fingerprint for your dataset. If even one
    /// value changes, the hash will be different. This helps verify data integrity.
    /// </remarks>
    /// <param name="dataPath">Path to the dataset.</param>
    /// <returns>The computed hash.</returns>
    string ComputeDatasetHash(string dataPath);

    /// <summary>
    /// Verifies that a dataset hasn't been modified by comparing its hash.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">Version to verify.</param>
    /// <param name="currentDataPath">Current location of the data.</param>
    /// <returns>True if the data matches the version, false if modified.</returns>
    bool VerifyDatasetIntegrity(string datasetName, string versionHash, string currentDataPath);

    /// <summary>
    /// Links a dataset version to a model training run.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a record showing which dataset version was used
    /// to train which model, enabling full reproducibility.
    /// </remarks>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">Version of the dataset.</param>
    /// <param name="runId">ID of the training run or experiment.</param>
    /// <param name="modelId">ID of the model that was trained.</param>
    void LinkDatasetToRun(string datasetName, string versionHash, string runId, string? modelId = null);

    /// <summary>
    /// Gets all training runs that used a specific dataset version.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">Version hash.</param>
    /// <returns>List of run IDs that used this dataset version.</returns>
    List<string> GetRunsUsingDataset(string datasetName, string versionHash);

    /// <summary>
    /// Gets the dataset version used by a specific training run.
    /// </summary>
    /// <param name="runId">ID of the training run.</param>
    /// <returns>Information about the dataset version used.</returns>
    DatasetVersion<T> GetDatasetForRun(string runId);

    /// <summary>
    /// Tags a dataset version for easy reference.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Tags are like bookmarks - they let you give a version a memorable name
    /// like "production-data" or "v2-cleaned" instead of using the hash.
    /// </remarks>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">Version to tag.</param>
    /// <param name="tag">The tag name to assign.</param>
    void TagDatasetVersion(string datasetName, string versionHash, string tag);

    /// <summary>
    /// Gets a dataset version by its tag.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="tag">The tag to look up.</param>
    /// <returns>The dataset version with that tag.</returns>
    DatasetVersion<T> GetDatasetByTag(string datasetName, string tag);

    /// <summary>
    /// Compares two dataset versions to see what changed.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="version1Hash">First version hash.</param>
    /// <param name="version2Hash">Second version hash.</param>
    /// <returns>Comparison showing differences between versions.</returns>
    DatasetComparison<T> CompareDatasetVersions(string datasetName, string version1Hash, string version2Hash);

    /// <summary>
    /// Records metadata about how a dataset was created or transformed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Lineage tracks the "family history" of your dataset - where it came from,
    /// what preprocessing was applied, etc. This is crucial for understanding and reproducing your work.
    /// </remarks>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">Version of the dataset.</param>
    /// <param name="lineage">Lineage information (source datasets, transformations applied).</param>
    void RecordDatasetLineage(string datasetName, string versionHash, DatasetLineage lineage);

    /// <summary>
    /// Gets the lineage information for a dataset version.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">Version hash.</param>
    /// <returns>Lineage information showing how the dataset was created.</returns>
    DatasetLineage GetDatasetLineage(string datasetName, string versionHash);

    /// <summary>
    /// Deletes a specific dataset version.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">Version to delete.</param>
    void DeleteDatasetVersion(string datasetName, string versionHash);

    /// <summary>
    /// Gets statistics about a dataset version.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This provides summary statistics about the dataset like
    /// number of rows, columns, data types, and basic descriptive statistics.
    /// </remarks>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="versionHash">Version hash.</param>
    /// <returns>Statistical summary of the dataset.</returns>
    DatasetStatistics<T> GetDatasetStatistics(string datasetName, string versionHash);

    /// <summary>
    /// Creates a snapshot of multiple related datasets together.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This captures multiple datasets at once (like train, validation, and test sets)
    /// so you can reproduce experiments that use all of them together.
    /// </remarks>
    /// <param name="snapshotName">Name for the snapshot.</param>
    /// <param name="datasets">Dictionary mapping dataset names to their version hashes.</param>
    /// <param name="description">Description of the snapshot.</param>
    /// <returns>The unique identifier for the snapshot.</returns>
    string CreateDatasetSnapshot(
        string snapshotName,
        Dictionary<string, string> datasets,
        string? description = null);

    /// <summary>
    /// Retrieves a dataset snapshot.
    /// </summary>
    /// <param name="snapshotName">Name of the snapshot.</param>
    /// <returns>Information about all datasets in the snapshot.</returns>
    DatasetSnapshot GetDatasetSnapshot(string snapshotName);
}
