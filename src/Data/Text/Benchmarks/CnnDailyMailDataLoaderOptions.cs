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
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public int MaxArticleLength { get; set; } = 512;
    public int MaxSummaryLength { get; set; } = 128;
    public int VocabularySize { get; set; } = 30000;
    public int? MaxSamples { get; set; }
}
