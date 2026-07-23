using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the CelebA face attributes data loader.
/// </summary>
/// <remarks>
/// <para>
/// CelebA contains ~200K celebrity face images with 40 binary attribute annotations.
/// Standard benchmark for face attribute prediction and generation.
/// </para>
/// </remarks>
public sealed class CelebADataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Image width after resizing. Default is 64.</summary>
    public int ImageWidth { get; set; } = 64;
    /// <summary>Image height after resizing. Default is 64.</summary>
    public int ImageHeight { get; set; } = 64;
    /// <summary>Number of binary face attributes. Default is 40.</summary>
    public int NumAttributes { get; set; } = 40;
    /// <summary>Normalize pixel values to [0,1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageWidth <= 0) throw new ArgumentOutOfRangeException(nameof(ImageWidth), "ImageWidth must be positive.");
        if (ImageHeight <= 0) throw new ArgumentOutOfRangeException(nameof(ImageHeight), "ImageHeight must be positive.");
        if (NumAttributes <= 0) throw new ArgumentOutOfRangeException(nameof(NumAttributes), "NumAttributes must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
