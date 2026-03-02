using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the WikiText-103 data loader.
/// </summary>
/// <remarks>
/// <para>
/// WikiText-103 is a large word-level language modeling dataset with over 100M tokens
/// from verified Good and Featured Wikipedia articles. Used for language model training/evaluation.
/// </para>
/// </remarks>
public sealed class WikiText103DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Sequence length for language modeling (context window). Default is 256.</summary>
    public int SequenceLength { get; set; } = 256;
    /// <summary>Maximum vocabulary size. Default is 30000.</summary>
    public int VocabularySize { get; set; } = 30000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
