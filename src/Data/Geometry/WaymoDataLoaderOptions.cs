namespace AiDotNet.Data.Geometry;

/// <summary>
/// Configuration options for the Waymo Open Dataset data loader.
/// </summary>
/// <remarks>
/// <para>
/// Waymo Open Dataset contains high-quality LiDAR and camera data from autonomous driving.
/// Includes 3D bounding boxes for vehicles, pedestrians, and cyclists.
/// </para>
/// </remarks>
public sealed class WaymoDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Number of points per sample. Default is 65536.</summary>
    public int PointsPerSample { get; set; } = 65536;
    /// <summary>Include intensity as 4th channel. Default is true.</summary>
    public bool IncludeIntensity { get; set; } = true;
}
