using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the MNIST data loader.
/// </summary>
public sealed class MnistDataLoaderOptions
{
    /// <summary>
    /// Dataset split to load. Default is Train.
    /// </summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>
    /// Root data path. When null, uses the default cache path (~/.aidotnet/datasets/mnist/).
    /// </summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download the dataset if not present. Default is true.
    /// </summary>
    public bool AutoDownload { get; set; } = true;

    /// <summary>
    /// Whether to normalize pixel values to [0, 1]. Default is true.
    /// </summary>
    public bool Normalize { get; set; } = true;

    /// <summary>
    /// Whether to flatten images to 1D vectors (784) instead of the spatial layout
    /// (<c>[B, 28, 28, 1]</c> NHWC or <c>[B, 1, 28, 28]</c> NCHW). Default is false.
    /// When true, <see cref="Layout"/> is ignored.
    /// </summary>
    public bool Flatten { get; set; }

    /// <summary>
    /// Axis ordering for the image tensor. Default is <see cref="ImageTensorLayout.NHWC"/>
    /// (<c>[B, 28, 28, 1]</c>). Set to <see cref="ImageTensorLayout.NCHW"/> for
    /// <c>[B, 1, 28, 28]</c>, which is the convention used by
    /// <c>ConvolutionalLayer&lt;T&gt;</c> and PyTorch-style models.
    /// Ignored when <see cref="Flatten"/> is true.
    /// </summary>
    public ImageTensorLayout Layout { get; set; } = ImageTensorLayout.NHWC;

    /// <summary>
    /// Optional maximum number of samples to load.
    /// </summary>
    public int? MaxSamples { get; set; }
}
