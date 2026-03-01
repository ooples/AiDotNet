using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the BigEarthNet data loader.
/// </summary>
/// <remarks>
/// <para>
/// BigEarthNet is a large-scale multi-label remote sensing dataset consisting of 590,326 Sentinel-2
/// image patches from 10 European countries. Each patch is 120x120 pixels with 12 spectral bands.
/// The multi-label classification task uses 19 CORINE Land Cover classes (BigEarthNet-19) or
/// 43 original CLC classes.
/// </para>
/// </remarks>
public sealed class BigEarthNetDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>
    /// Number of spectral bands to load. Default is 3 (RGB only).
    /// Set to 12 for all Sentinel-2 bands.
    /// </summary>
    public int NumBands { get; set; } = 3;

    /// <summary>
    /// Use the simplified 19-class label scheme (BigEarthNet-19). Default is true.
    /// Set to false for the original 43-class CLC labels.
    /// </summary>
    public bool Use19ClassScheme { get; set; } = true;
}
