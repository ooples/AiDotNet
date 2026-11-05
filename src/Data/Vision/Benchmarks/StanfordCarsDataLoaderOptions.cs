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
    /// <summary>Gets or sets which split of the dataset is loaded.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Gets or sets the dataset root directory; null uses the default data directory for this dataset.</summary>
    public string? DataPath { get; set; }
    /// <summary>Auto-download is OFF by default — Stanford URLs are unstable. Place archives manually.</summary>
    public bool AutoDownload { get; set; } = false;
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
