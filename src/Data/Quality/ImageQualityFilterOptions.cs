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
}
