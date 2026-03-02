using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the ImageNet-1K (ILSVRC 2012) data loader.
/// </summary>
/// <remarks>
/// <para>
/// ImageNet-1K contains ~1.28M training images and 50K validation images across 1,000 object categories.
/// Due to its large size (~150GB), auto-download is disabled by default. Provide the data path
/// to your local copy of the dataset.
/// </para>
/// </remarks>
public sealed class ImageNet1kDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download if not present. Default is false (dataset is ~150GB).
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load. Highly recommended for large datasets.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size (images are resized to this square dimension). Default is 224.</summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(ImageSize), "ImageSize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
