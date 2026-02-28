using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Open Images V7 data loader.
/// </summary>
/// <remarks>
/// <para>
/// Open Images V7 is a large-scale dataset with ~9M images and bounding box annotations
/// for 600 object categories. Annotations are provided as CSV files. Due to its massive size,
/// auto-download is disabled by default.
/// </para>
/// </remarks>
public sealed class OpenImagesDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download if not present. Default is false (dataset is very large).
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load. Highly recommended.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size. Default is 640.</summary>
    public int ImageSize { get; set; } = 640;

    /// <summary>
    /// Maximum number of detections per image. Detections are zero-padded to this size.
    /// Default is 100.
    /// </summary>
    public int MaxDetections { get; set; } = 100;
}
