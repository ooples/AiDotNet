using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the CNN/DailyMail summarization loader (Hermann et al. 2015 / See et al. 2017).
/// </summary>
/// <remarks>
/// <para>
/// CNN/DailyMail v3.0.0 — 287k train / 13.4k val / 11.5k test article-summary
/// pairs. The canonical English news abstractive-summarization benchmark.
/// AutoDownload pulls the HuggingFace parquet shards (abisee/cnn_dailymail).
/// </para>
/// </remarks>
public sealed class CnnDailyMailDataLoaderOptions
{
    /// <summary>Gets or sets which split of the dataset is loaded.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Gets or sets the dataset root directory; null uses the default data directory for this dataset.</summary>
    public string? DataPath { get; set; }
    /// <summary>Gets or sets whether the dataset is downloaded into the data directory when it is not there.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Gets or sets the token length each article is padded or truncated to.</summary>
    public int MaxArticleLength { get; set; } = 512;
    /// <summary>Gets or sets the token length each summary is padded or truncated to.</summary>
    public int MaxSummaryLength { get; set; } = 128;
    /// <summary>Gets or sets the most distinct tokens in the vocabulary built from the loaded text.</summary>
    public int VocabularySize { get; set; } = 30000;
    /// <summary>Gets or sets the most samples loaded; null loads the whole split.</summary>
    public int? MaxSamples { get; set; }
}
