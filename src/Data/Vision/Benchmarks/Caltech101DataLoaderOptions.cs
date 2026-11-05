using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Caltech-101 image classification data loader (Fei-Fei et al. 2004).
/// </summary>
/// <remarks>
/// <para>
/// Caltech-101 contains ≈ 9,000 images across 101 object categories plus a
/// background "BACKGROUND_Google" class (102 total). Image counts per
/// category vary from 40 to 800. Pre-CNN era benchmark, still used for
/// few-shot studies. Standard practice samples ≤ 30 images/class for
/// training and uses the rest for testing.
/// </para>
/// </remarks>
public sealed class Caltech101DataLoaderOptions
{
    /// <summary>Gets or sets which split of the dataset is loaded.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Gets or sets the dataset root directory; null uses the default data directory for this dataset.</summary>
    public string? DataPath { get; set; }
    /// <summary>Gets or sets whether the dataset is downloaded into the data directory when it is not there.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Gets or sets the side length, in pixels, images are resized to.</summary>
    public int ImageSize { get; set; } = 224;
    /// <summary>Gets or sets whether pixel values are scaled to [0, 1].</summary>
    public bool Normalize { get; set; } = true;
    /// <summary>Per-class images for the training split. Default 30 (standard).</summary>
    public int TrainImagesPerClass { get; set; } = 30;
    /// <summary>Gets or sets the most samples loaded; null loads the whole split.</summary>
    public int? MaxSamples { get; set; }
}
