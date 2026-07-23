using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the STL-10 image classification dataset (Coates et al. 2011).
/// </summary>
/// <remarks>
/// <para>
/// STL-10 is a 10-class, 96×96 RGB image classification benchmark with 500
/// labeled train + 800 test images per class, plus 100,000 unlabeled images
/// for self-supervised pretraining. Used widely for SSL/pretraining studies
/// since the unlabeled split is large compared to the small labeled split.
/// </para>
/// </remarks>
public sealed class Stl10DataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    /// <summary>
    /// When true, returns the unlabeled split (100k images) instead of train/test.
    /// Labels for unlabeled samples are zero-vectors. Default false.
    /// </summary>
    public bool UseUnlabeled { get; set; } = false;
    /// <summary>Normalize byte pixel values to [0, 1] when true (default), or keep raw 0..255 when false.</summary>
    public bool Normalize { get; set; } = true;
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
