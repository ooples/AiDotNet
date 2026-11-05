using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the SVHN (Street View House Numbers) loader (Netzer et al. 2011).
/// </summary>
/// <remarks>
/// <para>
/// SVHN Format-2 (cropped digits) — 32×32 RGB digit classification, 73,257
/// train + 26,032 test + 531,131 extra. The "harder than MNIST" baseline,
/// used widely for early CNN comparison studies and SSL ablations.
/// </para>
/// </remarks>
public sealed class SvhnDataLoaderOptions
{
    /// <summary>Gets or sets which split of the dataset is loaded.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Gets or sets the dataset root directory; null uses the default data directory for this dataset.</summary>
    public string? DataPath { get; set; }
    /// <summary>Gets or sets whether the dataset is downloaded into the data directory when it is not there.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Gets or sets whether pixel values are scaled to [0, 1].</summary>
    public bool Normalize { get; set; } = true;
    /// <summary>Include the 531k "extra" samples in the train split. Default false.</summary>
    public bool IncludeExtra { get; set; } = false;
    /// <summary>Gets or sets the most samples loaded; null loads the whole split.</summary>
    public int? MaxSamples { get; set; }
}
