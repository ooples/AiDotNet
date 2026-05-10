using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the XSum extreme summarization loader (Narayan et al. 2018).
/// </summary>
/// <remarks>
/// <para>
/// XSum — 226k BBC articles each paired with a single-sentence summary.
/// The "extreme" abstractive summarization benchmark; targets are highly
/// compressive (≈ 1:30 compression ratio). Splits: 204k train / 11.3k val / 11.3k test.
/// AutoDownload pulls a HuggingFace parquet shard.
/// </para>
/// </remarks>
public sealed class XSumDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public int MaxDocumentLength { get; set; } = 512;
    public int MaxSummaryLength { get; set; } = 64;
    public int VocabularySize { get; set; } = 30000;
    public int? MaxSamples { get; set; }
}
