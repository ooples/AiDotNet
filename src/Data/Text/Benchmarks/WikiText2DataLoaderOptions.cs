using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the WikiText-2 data loader.
/// </summary>
/// <remarks>
/// <para>
/// WikiText-2 is the small-scale word-level language modeling dataset
/// (≈ 2M tokens train, ≈ 245k tokens validation, ≈ 281k tokens test) drawn
/// from verified Good and Featured Wikipedia articles. It's the canonical
/// fast-iteration benchmark in the LM literature — papers commonly report
/// WikiText-2 perplexity first (cheap), then WikiText-103 second (slow, 50× larger).
/// </para>
/// <para>
/// Defaults are tuned for the smaller corpus: SequenceLength=32 keeps memory
/// footprint modest for fast research iteration, VocabularySize=16000 covers
/// roughly the 95th-percentile token-frequency cutoff at this scale. Override
/// these for paper-grade evaluation runs.
/// </para>
/// </remarks>
public sealed class WikiText2DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Sequence length for language modeling (context window). Default is 32 — tuned for fast iteration on the 2M-token corpus.</summary>
    public int SequenceLength { get; set; } = 32;
    /// <summary>Maximum vocabulary size. Default is 16000 — covers ~95th-percentile token-frequency cutoff at WikiText-2 scale.</summary>
    public int VocabularySize { get; set; } = 16000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
