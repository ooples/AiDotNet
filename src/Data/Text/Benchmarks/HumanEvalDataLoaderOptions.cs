using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the HumanEval Python code-generation
/// benchmark (Chen et al. 2021).
/// </summary>
/// <remarks>
/// <para>
/// HumanEval is a 164-problem hand-curated Python function-completion
/// benchmark. Each problem provides a function signature + docstring as
/// the prompt; the model completes the body. Pass@k scores against the
/// canonical unit tests are the standard metric. Used as the default
/// code-generation benchmark since GPT-3.5/Codex.
/// </para>
/// </remarks>
public sealed class HumanEvalDataLoaderOptions
{
    /// <summary>Dataset split to load. HumanEval has only Test (164 problems).</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Test;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum prompt length in tokens. Default is 256.</summary>
    public int MaxPromptLength { get; set; } = 256;
    /// <summary>Maximum solution length in tokens. Default is 256.</summary>
    public int MaxSolutionLength { get; set; } = 256;
    /// <summary>Maximum vocabulary size. Default is 8000 — code has a small effective vocabulary.</summary>
    public int VocabularySize { get; set; } = 8000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
