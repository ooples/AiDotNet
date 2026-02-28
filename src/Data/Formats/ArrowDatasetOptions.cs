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
}
