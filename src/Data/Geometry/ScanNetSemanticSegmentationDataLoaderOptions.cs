namespace AiDotNet.Data.Geometry;

/// <summary>
/// Configuration options for the ScanNet semantic segmentation data loader.
/// </summary>
public sealed class ScanNetSemanticSegmentationDataLoaderOptions
{
    public ScanNetSemanticSegmentationDataLoaderOptions()
    {
        Split = DatasetSplit.Train;
        PointsPerSample = 8192;
        IncludeColors = true;
        IncludeNormals = false;
        NormalizeColors = true;
        AutoDownload = false;
        InputFormat = ScanNetInputFormat.Auto;
        LabelMode = ScanNetLabelMode.Train20;
        IncludeUnknownClass = true;
        SamplingStrategy = PointSamplingStrategy.Random;
        PaddingStrategy = PointPaddingStrategy.Repeat;
        AutoDetectLabelColumn = true;
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
    /// Whether to include RGB colors when available.
    /// </summary>
    public bool IncludeColors { get; set; }

    /// <summary>
    /// Whether to include normals when available.
    /// </summary>
    public bool IncludeNormals { get; set; }

    /// <summary>
    /// Whether to normalize colors from 0-255 to 0-1.
    /// </summary>
    public bool NormalizeColors { get; set; }

    /// <summary>
    /// Root data path. When null, uses the default cache path.
    /// </summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download the dataset if not present.
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>
    /// Input data format selection.
    /// </summary>
    public ScanNetInputFormat InputFormat { get; set; }

    /// <summary>
    /// Label mapping mode.
    /// </summary>
    public ScanNetLabelMode LabelMode { get; set; }

    /// <summary>
    /// Whether to reserve an explicit unknown class at index 0.
    /// </summary>
    public bool IncludeUnknownClass { get; set; }

    /// <summary>
    /// Optional cap on the number of samples to load.
    /// </summary>
    public int? MaxSamples { get; set; }

    /// <summary>
    /// Optional random seed for reproducible sampling.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Strategy for sampling points from each scene.
    /// </summary>
    public PointSamplingStrategy SamplingStrategy { get; set; }

    /// <summary>
    /// Strategy for padding when fewer points exist than requested.
    /// </summary>
    public PointPaddingStrategy PaddingStrategy { get; set; }

    /// <summary>
    /// Whether to auto-detect a label column in preprocessed text files.
    /// </summary>
    public bool AutoDetectLabelColumn { get; set; }
}
