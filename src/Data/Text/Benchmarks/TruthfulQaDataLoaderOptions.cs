using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration for the TruthfulQA benchmark loader (Lin et al. 2022).
/// </summary>
/// <remarks>
/// <para>
/// TruthfulQA tests whether language models avoid common false beliefs in
/// 38 categories — 817 questions, each with a best correct answer and
/// multiple correct/incorrect distractors. This loader exposes the
/// generation-style version (best_answer as target).
/// </para>
/// </remarks>
public sealed class TruthfulQaDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Test;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public int MaxQuestionLength { get; set; } = 128;
    public int MaxAnswerLength { get; set; } = 128;
    public int VocabularySize { get; set; } = 16000;
    public int? MaxSamples { get; set; }
}
