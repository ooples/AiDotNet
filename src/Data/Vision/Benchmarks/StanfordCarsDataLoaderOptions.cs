using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration for the Stanford Cars dataset loader (Krause et al. 2013).
/// </summary>
/// <remarks>
/// <para>
/// Stanford Cars — 196 fine-grained car-model classes, 16,185 images.
/// Standard fine-grained classification benchmark. Original Stanford URLs
/// have been intermittent over the years, so <see cref="AutoDownload"/>
/// defaults to false; download manually from the various community mirrors
/// or HuggingFace and extract under <see cref="DataPath"/>.
/// </para>
/// </remarks>
public sealed class StanfordCarsDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    /// <summary>Auto-download is OFF by default — Stanford URLs are unstable. Place archives manually.</summary>
    public bool AutoDownload { get; set; } = false;
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
