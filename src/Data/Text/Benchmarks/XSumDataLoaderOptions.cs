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
    /// <summary>Gets or sets which split of the dataset is loaded.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Gets or sets the dataset root directory; null uses the default data directory for this dataset.</summary>
    public string? DataPath { get; set; }
    /// <summary>Gets or sets whether the dataset is downloaded into the data directory when it is not there.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Gets or sets the token length each document is padded or truncated to.</summary>
    public int MaxDocumentLength { get; set; } = 512;
    /// <summary>Gets or sets the token length each summary is padded or truncated to.</summary>
    public int MaxSummaryLength { get; set; } = 64;
    /// <summary>Gets or sets the most distinct tokens in the vocabulary built from the loaded text.</summary>
    public int VocabularySize { get; set; } = 30000;
    /// <summary>Gets or sets the most samples loaded; null loads the whole split.</summary>
    public int? MaxSamples { get; set; }
}
