using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Describable Textures Dataset (DTD) loader (Cimpoi et al. 2014).
/// </summary>
/// <remarks>
/// <para>
/// DTD — 47 texture classes × 120 images each (5,640 total). The canonical
/// texture-classification benchmark. Ten predefined train/val/test splits;
/// this loader uses split #1 by default (matches the standard reporting
/// convention). Image-list-based labels via <c>labels/{train,val,test}1.txt</c>.
/// </para>
/// </remarks>
public sealed class DtdDataLoaderOptions
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
    /// <summary>Predefined split index (1..10). Default 1.</summary>
    public int SplitIndex { get; set; } = 1;
    /// <summary>Gets or sets the most samples loaded; null loads the whole split.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(ImageSize), "ImageSize must be positive.");
        if (SplitIndex < 1 || SplitIndex > 10) throw new ArgumentOutOfRangeException(nameof(SplitIndex), "SplitIndex must be in [1, 10].");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
