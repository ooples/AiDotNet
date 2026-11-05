using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Oxford Flowers-102 dataset (Nilsback &amp; Zisserman 2008).
/// </summary>
/// <remarks>
/// <para>
/// Flowers-102 — 102 fine-grained flower species, 8,189 images total.
/// Standard fine-grained classification benchmark. The <c>setid.mat</c>
/// file defines the canonical 1,020/1,020/6,149 train/val/test split.
/// </para>
/// </remarks>
public sealed class Flowers102DataLoaderOptions
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

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(ImageSize), "ImageSize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
