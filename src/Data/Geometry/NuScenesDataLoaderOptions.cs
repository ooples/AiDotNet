namespace AiDotNet.Data.Geometry;

/// <summary>
/// Configuration options for the nuScenes data loader.
/// </summary>
/// <remarks>
/// <para>
/// nuScenes is a large-scale autonomous driving dataset with LiDAR, camera, and radar data.
/// Contains 1000 scenes with full 3D bounding box annotations for 23 object classes.
/// </para>
/// </remarks>
public sealed class NuScenesDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Number of points per sample. Default is 32768.</summary>
    public int PointsPerSample { get; set; } = 32768;
    /// <summary>Include intensity as 4th channel. Default is true.</summary>
    public bool IncludeIntensity { get; set; } = true;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (PointsPerSample <= 0) throw new ArgumentOutOfRangeException(nameof(PointsPerSample), "PointsPerSample must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
