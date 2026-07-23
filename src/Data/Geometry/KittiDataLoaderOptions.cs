namespace AiDotNet.Data.Geometry;

/// <summary>
/// Configuration options for the KITTI 3D object detection data loader.
/// </summary>
/// <remarks>
/// <para>
/// KITTI contains LiDAR point clouds from autonomous driving scenarios with 3D bounding box annotations.
/// Point clouds are stored as binary files (4 floats per point: x, y, z, reflectance).
/// </para>
/// </remarks>
public sealed class KittiDataLoaderOptions
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
    /// <summary>Include reflectance as 4th channel. Default is true.</summary>
    public bool IncludeReflectance { get; set; } = true;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (PointsPerSample <= 0) throw new ArgumentOutOfRangeException(nameof(PointsPerSample), "PointsPerSample must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
