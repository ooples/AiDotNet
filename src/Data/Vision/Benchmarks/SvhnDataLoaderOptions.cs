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
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public bool Normalize { get; set; } = true;
    /// <summary>Include the 531k "extra" samples in the train split. Default false.</summary>
    public bool IncludeExtra { get; set; } = false;
    public int? MaxSamples { get; set; }
}
