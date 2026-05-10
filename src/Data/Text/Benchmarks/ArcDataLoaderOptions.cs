using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Variant of the ARC benchmark to load.
/// </summary>
public enum ArcVariant
{
    /// <summary>ARC-Easy: simpler grade-school multiple-choice science questions.</summary>
    Easy,
    /// <summary>ARC-Challenge: harder questions that retrieval and PMI baselines fail.</summary>
    Challenge
}

/// <summary>
/// Configuration options for the AI2 Reasoning Challenge (ARC) data loader.
/// </summary>
/// <remarks>
/// <para>
/// ARC (Clark et al. 2018) is a 4-way multiple-choice grade-school science
/// QA benchmark. The Challenge subset contains questions that simple
/// retrieval/co-occurrence baselines fail; Easy subset is simpler. Both
/// are standard 0-shot LM-eval components.
/// </para>
/// </remarks>
public sealed class ArcDataLoaderOptions
{
    /// <summary>Which ARC variant: Easy or Challenge. Default Challenge.</summary>
    public ArcVariant Variant { get; set; } = ArcVariant.Challenge;
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum (question + choice) sequence length in tokens. Default is 128.</summary>
    public int MaxSequenceLength { get; set; } = 128;
    /// <summary>Maximum vocabulary size. Default is 16000.</summary>
    public int VocabularySize { get; set; } = 16000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
