namespace AiDotNet.Data.Formats;

/// <summary>
/// Configuration options for Apache Arrow-based dataset access.
/// </summary>
public sealed class ArrowDatasetOptions
{
    /// <summary>Path to the Arrow IPC file or directory of files. Required.</summary>
    public string DataPath { get; set; } = "";
    /// <summary>Name of the feature column. Default is "features".</summary>
    public string FeatureColumn { get; set; } = "features";
    /// <summary>Name of the label column. Default is "label".</summary>
    public string LabelColumn { get; set; } = "label";
    /// <summary>Whether to memory-map the file for large datasets. Default is true.</summary>
    public bool MemoryMap { get; set; } = true;
    /// <summary>Number of rows per batch when reading. Default is 1024.</summary>
    public int BatchSize { get; set; } = 1024;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(DataPath)) throw new ArgumentException("DataPath must not be empty.", nameof(DataPath));
        if (string.IsNullOrWhiteSpace(FeatureColumn)) throw new ArgumentException("FeatureColumn must not be empty.", nameof(FeatureColumn));
        if (string.IsNullOrWhiteSpace(LabelColumn)) throw new ArgumentException("LabelColumn must not be empty.", nameof(LabelColumn));
        if (BatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(BatchSize), "BatchSize must be positive.");
    }
}
