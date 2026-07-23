namespace AiDotNet.Data.Geometry;

/// <summary>
/// Configuration options for the SemanticKITTI data loader.
/// </summary>
/// <remarks>
/// <para>
/// SemanticKITTI provides per-point semantic labels for the KITTI Odometry benchmark.
/// 28 semantic classes for LiDAR point cloud segmentation.
/// </para>
/// </remarks>
public sealed class SemanticKittiDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Number of points per sample. Default is 16384.</summary>
    public int PointsPerSample { get; set; } = 16384;
    /// <summary>Number of semantic classes. Default is 28.</summary>
    public int NumClasses { get; set; } = 28;
}
