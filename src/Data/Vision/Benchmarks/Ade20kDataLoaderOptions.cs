using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the ADE20K semantic segmentation data loader.
/// </summary>
/// <remarks>
/// <para>
/// ADE20K contains ~25K images with per-pixel semantic annotations across 150 categories.
/// Standard benchmark for semantic segmentation models.
/// </para>
/// </remarks>
public sealed class Ade20kDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Image width after resizing. Default is 224.</summary>
    public int ImageWidth { get; set; } = 224;
    /// <summary>Image height after resizing. Default is 224.</summary>
    public int ImageHeight { get; set; } = 224;
    /// <summary>Number of semantic classes. Default is 150.</summary>
    public int NumClasses { get; set; } = 150;
    /// <summary>Normalize pixel values to [0,1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageWidth <= 0) throw new ArgumentOutOfRangeException(nameof(ImageWidth), "ImageWidth must be positive.");
        if (ImageHeight <= 0) throw new ArgumentOutOfRangeException(nameof(ImageHeight), "ImageHeight must be positive.");
        if (NumClasses <= 0) throw new ArgumentOutOfRangeException(nameof(NumClasses), "NumClasses must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
