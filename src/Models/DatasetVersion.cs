namespace AiDotNet.Models;

/// <summary>
/// Represents a versioned dataset with integrity verification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class DatasetVersion<T>
{
    /// <summary>
    /// Gets or sets the unique identifier for this version.
    /// </summary>
    public string VersionId { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the dataset name.
    /// </summary>
    public string DatasetName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version number.
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets the hash for integrity verification.
    /// </summary>
    public string? Hash { get; set; }

    /// <summary>
    /// Gets or sets the storage path.
    /// </summary>
    public string? StoragePath { get; set; }

    /// <summary>
    /// Gets or sets when this version was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the tags.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets the number of records.
    /// </summary>
    public long RecordCount { get; set; }

    /// <summary>
    /// Gets or sets the size in bytes.
    /// </summary>
    public long SizeBytes { get; set; }
}

/// <summary>
/// Information about a dataset version.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class DatasetVersionInfo<T>
{
    /// <summary>
    /// Gets or sets the version ID.
    /// </summary>
    public string VersionId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version number.
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets the creation time.
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Gets or sets the hash.
    /// </summary>
    public string? Hash { get; set; }

    /// <summary>
    /// Gets or sets the record count.
    /// </summary>
    public long RecordCount { get; set; }

    /// <summary>
    /// Gets or sets the size in bytes.
    /// </summary>
    public long SizeBytes { get; set; }

    /// <summary>
    /// Gets or sets the description.
    /// </summary>
    public string? Description { get; set; }
}

/// <summary>
/// Comparison results between two dataset versions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class DatasetComparison<T>
{
    /// <summary>
    /// Gets or sets the first version.
    /// </summary>
    public int Version1 { get; set; }

    /// <summary>
    /// Gets or sets the second version.
    /// </summary>
    public int Version2 { get; set; }

    /// <summary>
    /// Gets or sets the number of added records.
    /// </summary>
    public long RecordsAdded { get; set; }

    /// <summary>
    /// Gets or sets the number of removed records.
    /// </summary>
    public long RecordsRemoved { get; set; }

    /// <summary>
    /// Gets or sets the number of modified records.
    /// </summary>
    public long RecordsModified { get; set; }

    /// <summary>
    /// Gets or sets the schema changes.
    /// </summary>
    public List<string> SchemaChanges { get; set; } = new();

    /// <summary>
    /// Gets or sets the statistical changes.
    /// </summary>
    public Dictionary<string, (T OldValue, T NewValue)> StatisticalChanges { get; set; } = new();
}

/// <summary>
/// Lineage information for a dataset showing its origin and transformations.
/// </summary>
public class DatasetLineage
{
    /// <summary>
    /// Gets or sets the dataset name.
    /// </summary>
    public string DatasetName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version.
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets the source dataset (if derived).
    /// </summary>
    public string? SourceDataset { get; set; }

    /// <summary>
    /// Gets or sets the source version.
    /// </summary>
    public int? SourceVersion { get; set; }

    /// <summary>
    /// Gets or sets the transformations applied.
    /// </summary>
    public List<string> Transformations { get; set; } = new();

    /// <summary>
    /// Gets or sets the creation timestamp.
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Gets or sets the creator.
    /// </summary>
    public string? Creator { get; set; }

    /// <summary>
    /// Gets or sets training runs that used this dataset.
    /// </summary>
    public List<string> UsedInRuns { get; set; } = new();
}

/// <summary>
/// Statistical summary of a dataset.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class DatasetStatistics<T>
{
    /// <summary>
    /// Gets or sets the record count.
    /// </summary>
    public long RecordCount { get; set; }

    /// <summary>
    /// Gets or sets the column count.
    /// </summary>
    public int ColumnCount { get; set; }

    /// <summary>
    /// Gets or sets the missing values per column.
    /// </summary>
    public Dictionary<string, long> MissingValues { get; set; } = new();

    /// <summary>
    /// Gets or sets the numeric column statistics.
    /// </summary>
    public Dictionary<string, NumericColumnStats<T>> NumericStats { get; set; } = new();

    /// <summary>
    /// Gets or sets the categorical column statistics.
    /// </summary>
    public Dictionary<string, CategoricalColumnStats> CategoricalStats { get; set; } = new();
}

/// <summary>
/// Statistics for a numeric column.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public class NumericColumnStats<T>
{
    /// <summary>
    /// Gets or sets the minimum value.
    /// </summary>
    public T? Min { get; set; }

    /// <summary>
    /// Gets or sets the maximum value.
    /// </summary>
    public T? Max { get; set; }

    /// <summary>
    /// Gets or sets the mean value.
    /// </summary>
    public T? Mean { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation.
    /// </summary>
    public T? StdDev { get; set; }

    /// <summary>
    /// Gets or sets the median value.
    /// </summary>
    public T? Median { get; set; }
}

/// <summary>
/// Statistics for a categorical column.
/// </summary>
public class CategoricalColumnStats
{
    /// <summary>
    /// Gets or sets the number of unique values.
    /// </summary>
    public int UniqueCount { get; set; }

    /// <summary>
    /// Gets or sets the most frequent value.
    /// </summary>
    public string? MostFrequent { get; set; }

    /// <summary>
    /// Gets or sets the frequency of the most frequent value.
    /// </summary>
    public long MostFrequentCount { get; set; }

    /// <summary>
    /// Gets or sets the value distribution.
    /// </summary>
    public Dictionary<string, long> ValueCounts { get; set; } = new();
}

/// <summary>
/// A snapshot of a dataset at a point in time.
/// </summary>
public class DatasetSnapshot
{
    /// <summary>
    /// Gets or sets the snapshot ID.
    /// </summary>
    public string SnapshotId { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the dataset name.
    /// </summary>
    public string DatasetName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version at snapshot time.
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets when the snapshot was taken.
    /// </summary>
    public DateTime SnapshotTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the snapshot description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the storage path.
    /// </summary>
    public string? StoragePath { get; set; }

    /// <summary>
    /// Gets or sets the hash for integrity verification.
    /// </summary>
    public string? Hash { get; set; }
}
