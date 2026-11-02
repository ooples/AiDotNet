using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Food-101 image classification data loader.
/// </summary>
/// <remarks>
/// <para>
/// Food-101 (Bossard et al. 2014) — 101 food categories, 101,000 images
/// total (750 train + 250 test per class). Standard fine-grained
/// classification benchmark. Auto-downloads from the ETH Zürich mirror.
/// </para>
/// </remarks>
public sealed class Food101DataLoaderOptions
{
    /// <summary>Gets or sets which split of the dataset is loaded.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Gets or sets the dataset root directory; null uses the default data directory for this dataset.</summary>
    public string? DataPath { get; set; }
    /// <summary>Gets or sets whether the dataset is downloaded into the data directory when it is not there.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Gets or sets the side length, in pixels, images are resized to.</summary>
    public int ImageSize { get; set; } = 224;
    /// <summary>Gets or sets whether pixel values are scaled to [0, 1].</summary>
    public bool Normalize { get; set; } = true;
    /// <summary>Gets or sets the most samples loaded; null loads the whole split.</summary>
    public int? MaxSamples { get; set; }
}
