namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for image quality filtering.
/// </summary>
/// <remarks>
/// Filters images based on resolution, aspect ratio, and statistical quality metrics.
/// </remarks>
public sealed class ImageQualityFilterOptions
{
    /// <summary>Minimum image width in pixels. Default is 64.</summary>
    public int MinWidth { get; set; } = 64;
    /// <summary>Minimum image height in pixels. Default is 64.</summary>
    public int MinHeight { get; set; } = 64;
    /// <summary>Maximum allowed aspect ratio (width/height or height/width). Default is 5.0.</summary>
    public double MaxAspectRatio { get; set; } = 5.0;
    /// <summary>Minimum standard deviation of pixel values (detects blank images). Default is 5.0.</summary>
    public double MinPixelStdDev { get; set; } = 5.0;
    /// <summary>Maximum fraction of pixels with the same value (detects solid-color images). Default is 0.9.</summary>
    public double MaxDominantColorRatio { get; set; } = 0.9;
    /// <summary>Minimum number of unique pixel values. Default is 10.</summary>
    public int MinUniqueColors { get; set; } = 10;
    /// <summary>Minimum file size in bytes (filters corrupt/tiny files). Default is 1024.</summary>
    public long MinFileSize { get; set; } = 1024;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MinWidth <= 0) throw new ArgumentOutOfRangeException(nameof(MinWidth), "MinWidth must be positive.");
        if (MinHeight <= 0) throw new ArgumentOutOfRangeException(nameof(MinHeight), "MinHeight must be positive.");
        if (MaxAspectRatio <= 0) throw new ArgumentOutOfRangeException(nameof(MaxAspectRatio), "MaxAspectRatio must be positive.");
        if (MinPixelStdDev < 0) throw new ArgumentOutOfRangeException(nameof(MinPixelStdDev), "MinPixelStdDev must be non-negative.");
        if (MaxDominantColorRatio < 0 || MaxDominantColorRatio > 1) throw new ArgumentOutOfRangeException(nameof(MaxDominantColorRatio), "MaxDominantColorRatio must be between 0 and 1.");
        if (MinUniqueColors < 0) throw new ArgumentOutOfRangeException(nameof(MinUniqueColors), "MinUniqueColors must be non-negative.");
        if (MinFileSize < 0) throw new ArgumentOutOfRangeException(nameof(MinFileSize), "MinFileSize must be non-negative.");
    }
}
