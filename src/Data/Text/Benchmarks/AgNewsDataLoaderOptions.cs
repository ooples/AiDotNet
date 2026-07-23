using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the AG News topic classification data loader.
/// </summary>
/// <remarks>
/// <para>
/// AG News is a 4-class news topic classification dataset (World, Sports,
/// Business, Sci/Tech) — 120k training examples, 7.6k test examples. The
/// canonical small classification benchmark from Zhang et al. 2015.
/// </para>
/// </remarks>
public sealed class AgNewsDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum sequence length per example. Default is 128.</summary>
    public int MaxSequenceLength { get; set; } = 128;
    /// <summary>Maximum vocabulary size. Default is 30000.</summary>
    public int VocabularySize { get; set; } = 30000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
