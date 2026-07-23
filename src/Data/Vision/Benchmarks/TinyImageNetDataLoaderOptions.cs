using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Tiny ImageNet (200-class, 64×64) data loader.
/// </summary>
/// <remarks>
/// <para>
/// Tiny ImageNet is the standard middle-ground vision benchmark between
/// CIFAR-100 and full ImageNet — 200 classes with 500 training, 50
/// validation, 50 test images each at 64×64 resolution. Used widely for
/// architecture-search and few-shot studies. Produced by Stanford CS231n.
/// </para>
/// </remarks>
public sealed class TinyImageNetDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Image size (height = width). Default is 64.</summary>
    public int ImageSize { get; set; } = 64;
    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
