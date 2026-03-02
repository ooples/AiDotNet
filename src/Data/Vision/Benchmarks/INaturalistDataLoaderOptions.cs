using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the iNaturalist data loader.
/// </summary>
/// <remarks>
/// <para>
/// iNaturalist is a large-scale species classification dataset with fine-grained categories.
/// The 2021 version contains ~2.7M images across 10,000 species. The dataset exhibits
/// significant class imbalance (long-tailed distribution), making it useful for imbalanced learning research.
/// </para>
/// </remarks>
public sealed class INaturalistDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download if not present. Default is false (dataset is ~200GB).
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load. Highly recommended.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size. Default is 224.</summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>
    /// iNaturalist version year. Default is 2021.
    /// Available versions: 2017, 2018, 2019, 2021.
    /// </summary>
    public int Version { get; set; } = 2021;
}
