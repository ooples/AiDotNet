using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Places365 data loader.
/// </summary>
/// <remarks>
/// <para>
/// Places365 is a scene recognition dataset with 1.8M training images across 365 scene categories
/// (e.g., bedroom, kitchen, forest, highway). Available in standard (256x256) and challenge (high-res) versions.
/// </para>
/// </remarks>
public sealed class Places365DataLoaderOptions
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

    /// <summary>Target image size. Default is 256.</summary>
    public int ImageSize { get; set; } = 256;
}
