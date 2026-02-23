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
    /// Whether to flatten images to 1D vectors (784) instead of 2D (28x28). Default is false.
    /// </summary>
    public bool Flatten { get; set; }

    /// <summary>
    /// Optional maximum number of samples to load.
    /// </summary>
    public int? MaxSamples { get; set; }
}
