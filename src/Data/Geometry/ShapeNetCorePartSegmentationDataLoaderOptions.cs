namespace AiDotNet.Data.Geometry;

/// <summary>
/// Configuration options for the ShapeNetCore part segmentation data loader.
/// </summary>
public sealed class ShapeNetCorePartSegmentationDataLoaderOptions
{
    public ShapeNetCorePartSegmentationDataLoaderOptions()
    {
        Split = DatasetSplit.Train;
        PointsPerSample = 2048;
        IncludeNormals = true;
        AutoDownload = true;
        NumClasses = 50;
        SamplingStrategy = PointSamplingStrategy.Random;
        PaddingStrategy = PointPaddingStrategy.Repeat;
    }

    /// <summary>
    /// Dataset split to load.
    /// </summary>
    public DatasetSplit Split { get; set; }

    /// <summary>
    /// Number of points per sample.
    /// </summary>
    public int PointsPerSample { get; set; }

    /// <summary>
    /// Whether to include normals when available.
    /// </summary>
    public bool IncludeNormals { get; set; }

    /// <summary>
    /// Root data path. When null, uses the default cache path.
    /// </summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download the dataset if not present.
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>
    /// Number of part classes in the dataset.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Optional cap on the number of samples to load.
    /// </summary>
    public int? MaxSamples { get; set; }

    /// <summary>
    /// Optional random seed for reproducible sampling.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Strategy for sampling points from each shape.
    /// </summary>
    public PointSamplingStrategy SamplingStrategy { get; set; }

    /// <summary>
    /// Strategy for padding when fewer points exist than requested.
    /// </summary>
    public PointPaddingStrategy PaddingStrategy { get; set; }
}
