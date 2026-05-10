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
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public int ImageSize { get; set; } = 224;
    public bool Normalize { get; set; } = true;
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(ImageSize), "ImageSize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
