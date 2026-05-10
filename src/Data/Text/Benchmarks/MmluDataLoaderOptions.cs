using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the MMLU (Massive Multitask Language Understanding) loader (Hendrycks et al. 2021).
/// </summary>
/// <remarks>
/// <para>
/// MMLU — 57 subjects × ~270 multi-choice questions each, spanning STEM,
/// humanities, social sciences, professional, and other categories. The
/// canonical broad-knowledge LLM eval benchmark since GPT-3. 4-way
/// multiple choice. Splits: dev (5/subject, used for few-shot prompting),
/// val (~85/subject), test (~14k total).
/// </para>
/// </remarks>
public sealed class MmluDataLoaderOptions
{
    /// <summary>Dataset split. Default Test (the canonical eval split).</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Test;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional subject filter (case-insensitive substring). Null = all 57 subjects.</summary>
    public string? SubjectFilter { get; set; }
    /// <summary>Maximum encoded question + choice length in tokens. Default 256.</summary>
    public int MaxQuestionLength { get; set; } = 256;
    /// <summary>Maximum vocabulary size for the BPE-style tokenizer. Default 16000.</summary>
    public int VocabularySize { get; set; } = 16000;
    /// <summary>Optional maximum number of samples to load (for fast iteration / smoke testing).</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MaxQuestionLength <= 0) throw new ArgumentOutOfRangeException(nameof(MaxQuestionLength), "MaxQuestionLength must be positive.");
        if (VocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(VocabularySize), "VocabularySize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
