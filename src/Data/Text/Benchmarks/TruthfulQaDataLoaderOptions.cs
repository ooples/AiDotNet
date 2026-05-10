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

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MaxQuestionLength <= 0) throw new ArgumentOutOfRangeException(nameof(MaxQuestionLength), "MaxQuestionLength must be positive.");
        if (MaxAnswerLength <= 0) throw new ArgumentOutOfRangeException(nameof(MaxAnswerLength), "MaxAnswerLength must be positive.");
        if (VocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(VocabularySize), "VocabularySize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
