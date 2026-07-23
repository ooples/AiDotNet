using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the HellaSwag commonsense NLI benchmark
/// (Zellers et al. 2019).
/// </summary>
/// <remarks>
/// <para>
/// HellaSwag is a 4-way multiple-choice commonsense reasoning benchmark
/// where each example presents a context sentence and 4 possible endings;
/// only one is the natural continuation. ≈ 39,905 train / 10,042 val
/// problems. Adversarially filtered against early LMs — remains one of
/// the standard 0-shot LM-eval benchmarks today.
/// </para>
/// </remarks>
public sealed class HellaswagDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum sequence length per (context + ending) pair. Default is 256.</summary>
    public int MaxSequenceLength { get; set; } = 256;
    /// <summary>Maximum vocabulary size. Default is 16000.</summary>
    public int VocabularySize { get; set; } = 16000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
