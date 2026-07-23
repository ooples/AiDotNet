using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the CIFAR-10 data loader.
/// </summary>
public sealed class Cifar10DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;
    /// <summary>
    /// Axis ordering for the image tensor. Default is <see cref="ImageTensorLayout.NHWC"/>
    /// (<c>[B, 32, 32, 3]</c>). Set to <see cref="ImageTensorLayout.NCHW"/> for
    /// <c>[B, 3, 32, 32]</c>.
    /// </summary>
    public ImageTensorLayout Layout { get; set; } = ImageTensorLayout.NHWC;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
