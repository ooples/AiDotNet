using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the Mostly Basic Python Problems (MBPP) loader (Austin et al. 2021).
/// </summary>
/// <remarks>
/// <para>
/// MBPP is a 1,000-problem benchmark of basic Python programming tasks, each with
/// a natural-language description, canonical reference solution, and 3 unit tests.
/// Standard split: 974 train / 90 val / 500 test (with 100 reserved for prompt).
/// Used as the entry-level code-generation benchmark alongside HumanEval.
/// </para>
/// </remarks>
public sealed class MbppDataLoaderOptions
{
    /// <summary>Dataset split. Default Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Auto-download.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Max prompt length in tokens. Default 256.</summary>
    public int MaxPromptLength { get; set; } = 256;
    /// <summary>Max solution length in tokens. Default 256.</summary>
    public int MaxSolutionLength { get; set; } = 256;
    /// <summary>Max vocab. Default 8000.</summary>
    public int VocabularySize { get; set; } = 8000;
    /// <summary>Optional sample cap.</summary>
    public int? MaxSamples { get; set; }
}
