using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Oxford-IIIT Pet dataset loader (Parkhi et al. 2012).
/// </summary>
/// <remarks>
/// <para>
/// Oxford-IIIT Pets — 37 dog/cat breeds, ~200 images per breed (7,349 total).
/// Standard fine-grained classification benchmark with both species (binary)
/// and breed (37-way) labels. Filenames encode breeds: e.g.
/// <c>Abyssinian_100.jpg</c>.
/// </para>
/// </remarks>
public sealed class OxfordPetsDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public int ImageSize { get; set; } = 224;
    public bool Normalize { get; set; } = true;
    public int? MaxSamples { get; set; }
}
