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
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public int ImageSize { get; set; } = 224;
    public bool Normalize { get; set; } = true;
    /// <summary>Per-class images for the training split. Default 30 (standard).</summary>
    public int TrainImagesPerClass { get; set; } = 30;
    public int? MaxSamples { get; set; }
}
