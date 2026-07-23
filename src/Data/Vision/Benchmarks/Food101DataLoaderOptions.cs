using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Food-101 image classification data loader.
/// </summary>
/// <remarks>
/// <para>
/// Food-101 (Bossard et al. 2014) — 101 food categories, 101,000 images
/// total (750 train + 250 test per class). Standard fine-grained
/// classification benchmark. Auto-downloads from the ETH Zürich mirror.
/// </para>
/// </remarks>
public sealed class Food101DataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public int ImageSize { get; set; } = 224;
    public bool Normalize { get; set; } = true;
    public int? MaxSamples { get; set; }
}
