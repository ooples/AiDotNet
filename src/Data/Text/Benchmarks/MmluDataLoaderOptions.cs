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
    public DatasetSplit Split { get; set; } = DatasetSplit.Test;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional subject filter (case-insensitive substring). Null = all 57 subjects.</summary>
    public string? SubjectFilter { get; set; }
    public int MaxQuestionLength { get; set; } = 256;
    public int VocabularySize { get; set; } = 16000;
    public int? MaxSamples { get; set; }
}
