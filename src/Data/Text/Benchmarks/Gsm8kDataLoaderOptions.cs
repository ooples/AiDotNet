using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the GSM8K math word-problem benchmark.
/// </summary>
/// <remarks>
/// <para>
/// GSM8K (Cobbe et al. 2021) is the canonical grade-school math benchmark
/// for chain-of-thought reasoning evaluation. ≈ 7,473 train / 1,319 test
/// problems, each with a multi-step natural-language solution and a final
/// numerical answer prefixed by <c>####</c>.
/// </para>
/// </remarks>
public sealed class Gsm8kDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum question length in tokens. Default is 256.</summary>
    public int MaxQuestionLength { get; set; } = 256;
    /// <summary>Maximum answer length in tokens (full chain-of-thought). Default is 512.</summary>
    public int MaxAnswerLength { get; set; } = 512;
    /// <summary>Maximum vocabulary size. Default is 16000.</summary>
    public int VocabularySize { get; set; } = 16000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
