using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the Penn Treebank (PTB) data loader.
/// </summary>
/// <remarks>
/// <para>
/// PTB is the classic small-scale word-level LM benchmark — ≈ 887k tokens train,
/// ≈ 70k val, ≈ 79k test, vocab ≈ 10k after preprocessing. Most pre-2018 LM
/// papers report PTB perplexity. Useful as a sanity check before scaling.
/// Tokenization follows the Mikolov-style preprocessed split.
/// </para>
/// </remarks>
public sealed class PennTreebankDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Sequence length (BPTT context). Default is 35 — the original Mikolov setting.</summary>
    public int SequenceLength { get; set; } = 35;
    /// <summary>Maximum vocabulary size. Default is 10000 (matches the canonical preprocessed PTB).</summary>
    public int VocabularySize { get; set; } = 10000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
