using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Pascal VOC data loader.
/// </summary>
/// <remarks>
/// <para>
/// Pascal VOC (Visual Object Classes) is a classic object detection benchmark with 20 categories.
/// VOC2007 has 5K train/val + 5K test images. VOC2012 has 11.5K train/val images.
/// Annotations are in XML format (one per image).
/// </para>
/// </remarks>
public sealed class PascalVocDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size. Default is 500.</summary>
    public int ImageSize { get; set; } = 500;

    /// <summary>
    /// Maximum number of detections per image. Detections are zero-padded to this size.
    /// Default is 50.
    /// </summary>
    public int MaxDetections { get; set; } = 50;

    /// <summary>
    /// VOC year version. Default is "2012". Options: "2007", "2012".
    /// </summary>
    public string Year { get; set; } = "2012";
}
