using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Oxford-IIIT Pet dataset loader (Parkhi et al. 2012).
/// </summary>
/// <remarks>
/// <para>
/// Oxford-IIIT Pets — 37 dog/cat breeds, ~200 images per breed (7,349 total).
/// Standard fine-grained classification benchmark with both species (binary)
/// and breed (37-way) labels. Filenames encode breeds: e.g.
/// <c>Abyssinian_100.jpg</c>.
/// </para>
/// </remarks>
public sealed class OxfordPetsDataLoaderOptions
{
    /// <summary>Dataset split to load. Default Train (uses annotations/trainval.txt vs. test.txt).</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Target square image edge in pixels. Default 224 (ImageNet eval default).</summary>
    public int ImageSize { get; set; } = 224;
    /// <summary>Normalize byte pixel values to [0, 1] when true (default), or keep raw 0..255 when false.</summary>
    public bool Normalize { get; set; } = true;
    /// <summary>Optional maximum number of samples to load (for fast iteration / smoke testing).</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(ImageSize), "ImageSize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
