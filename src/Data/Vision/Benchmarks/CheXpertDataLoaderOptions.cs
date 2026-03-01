using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the CheXpert data loader.
/// </summary>
/// <remarks>
/// <para>
/// CheXpert (Chest eXpert) is a large chest radiograph dataset from Stanford with 224,316 chest X-rays
/// of 65,240 patients. It has 14 observation labels with explicit uncertainty handling:
/// positive (1), negative (0), uncertain (-1), or not mentioned (blank).
/// </para>
/// </remarks>
public sealed class CheXpertDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download if not present. Default is false (requires Stanford AIMI agreement).
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size. Default is 224.</summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>
    /// Policy for handling uncertain labels (-1). Default is Zeros.
    /// Zeros: treat uncertain as negative. Ones: treat as positive. Ignore: set to 0.
    /// </summary>
    public UncertaintyPolicy UncertaintyHandling { get; set; } = UncertaintyPolicy.Zeros;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(ImageSize), "ImageSize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
        if (!Enum.IsDefined(typeof(DatasetSplit), Split))
            throw new ArgumentOutOfRangeException(nameof(Split), "Split must be a valid DatasetSplit value.");
        if (!Enum.IsDefined(typeof(UncertaintyPolicy), UncertaintyHandling))
            throw new ArgumentOutOfRangeException(nameof(UncertaintyHandling), "UncertaintyHandling must be a valid UncertaintyPolicy value.");
        if (DataPath is not null && string.IsNullOrWhiteSpace(DataPath))
            throw new ArgumentException("DataPath must not be empty or whitespace when provided.", nameof(DataPath));
    }
}

/// <summary>
/// Policy for handling uncertain labels in CheXpert.
/// </summary>
public enum UncertaintyPolicy
{
    /// <summary>Treat uncertain labels as negative (0). Most conservative.</summary>
    Zeros,
    /// <summary>Treat uncertain labels as positive (1). Most sensitive.</summary>
    Ones,
    /// <summary>Exclude samples with uncertain labels from training entirely.</summary>
    Ignore
}
