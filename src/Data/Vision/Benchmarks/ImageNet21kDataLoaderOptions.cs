using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the ImageNet-21K data loader.
/// </summary>
/// <remarks>
/// <para>
/// ImageNet-21K contains ~14.2M images across 21,841 categories (the full ImageNet hierarchy).
/// Due to its massive size, auto-download is disabled by default.
/// </para>
/// </remarks>
public sealed class ImageNet21kDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download if not present. Default is false (dataset is ~1.3TB).
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load. Highly recommended for this dataset.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size (images are resized to this square dimension). Default is 224.</summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>
    /// Number of classes to load. Default is null (all 21,841 classes).
    /// Set to a lower value to load a subset of the hierarchy.
    /// </summary>
    public int? MaxClasses { get; set; }
}
