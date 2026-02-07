using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision;

/// <summary>
/// Configuration options for the <see cref="ImageFolderDataset{T}"/>.
/// </summary>
public sealed class ImageFolderDatasetOptions
{
    /// <summary>
    /// Root directory containing class subdirectories.
    /// </summary>
    public string RootDirectory { get; set; } = string.Empty;

    /// <summary>
    /// File extensions to include. Default includes formats supported by ImageHelper (BMP, PPM, PGM).
    /// </summary>
    public string[] Extensions { get; set; } = new[] { ".bmp", ".ppm", ".pgm" };

    /// <summary>
    /// Target image width in pixels. Images will be resized to this width.
    /// </summary>
    public int ImageWidth { get; set; } = 224;

    /// <summary>
    /// Target image height in pixels. Images will be resized to this height.
    /// </summary>
    public int ImageHeight { get; set; } = 224;

    /// <summary>
    /// Number of color channels. 1 for grayscale, 3 for RGB. Default is 3.
    /// </summary>
    public int Channels { get; set; } = 3;

    /// <summary>
    /// Whether to normalize pixel values to [0, 1]. Default is true.
    /// </summary>
    public bool NormalizePixels { get; set; } = true;

    /// <summary>
    /// Dataset split to load.
    /// </summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.All;

    /// <summary>
    /// Optional maximum number of samples to load.
    /// </summary>
    public int? MaxSamples { get; set; }

    /// <summary>
    /// Optional random seed for reproducible sampling.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Whether to search subdirectories recursively. Default is false (one level of class dirs).
    /// </summary>
    public bool Recursive { get; set; }
}
