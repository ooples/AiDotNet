using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the SQuAD data loader.
/// </summary>
/// <remarks>
/// <para>
/// SQuAD (Stanford Question Answering Dataset) contains 100K+ question-answer pairs
/// on Wikipedia articles. SQuAD 2.0 adds 50K unanswerable questions.
/// </para>
/// </remarks>
public sealed class SquadDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum context length in tokens. Default is 384.</summary>
    public int MaxContextLength { get; set; } = 384;
    /// <summary>Maximum question length in tokens. Default is 64.</summary>
    public int MaxQuestionLength { get; set; } = 64;
    /// <summary>Maximum vocabulary size. Default is 30000.</summary>
    public int VocabularySize { get; set; } = 30000;
    /// <summary>Use SQuAD 2.0 (includes unanswerable questions). Default is false.</summary>
    public bool Version2 { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
