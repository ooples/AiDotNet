using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Functional Map of the World (fMoW) data loader.
/// </summary>
/// <remarks>
/// <para>
/// fMoW is a satellite imagery dataset for temporal land use classification with 62 categories.
/// It contains over 1M images from 200+ countries with temporal metadata (images of the same
/// location taken at different times). This makes it useful for temporal analysis of land use change.
/// </para>
/// </remarks>
public sealed class FMoWDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download if not present. Default is false (dataset is very large).
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load. Highly recommended.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size. Default is 224.</summary>
    public int ImageSize { get; set; } = 224;
}
