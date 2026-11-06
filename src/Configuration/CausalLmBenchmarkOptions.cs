using System.Collections.Generic;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for causal language-model dataset suites (<see cref="Enums.BenchmarkSuite.LAMBADA"/>
/// and <see cref="Enums.BenchmarkSuite.Perplexity"/>).
/// </summary>
public sealed class CausalLmBenchmarkOptions
{
    /// <summary>
    /// Gets or sets an optional path to a LAMBADA JSONL (<c>{"text": "..."}</c> per line) or plain-text passage file.
    /// </summary>
    public string? LambadaDatasetPath { get; set; }

    /// <summary>
    /// Gets or sets optional in-memory full LAMBADA passages (each ending with the held-out target word).
    /// </summary>
    public IReadOnlyList<string>? LambadaPassages { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of subword tokens decoded per word on LAMBADA (default 8).
    /// </summary>
    public int MaxSubwordTokensPerWord { get; set; } = 8;

    /// <summary>
    /// Gets or sets an optional path to a plain-text evaluation corpus for <see cref="Enums.BenchmarkSuite.Perplexity"/>.
    /// </summary>
    public string? PerplexityCorpusPath { get; set; }

    /// <summary>
    /// Gets or sets optional in-memory evaluation text for <see cref="Enums.BenchmarkSuite.Perplexity"/>.
    /// </summary>
    public string? PerplexityCorpusText { get; set; }

    /// <summary>
    /// Gets or sets the non-overlapping context window length (in tokens) for perplexity evaluation (default 512).
    /// </summary>
    public int ContextWindowLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the starting token index within the tokenized corpus for held-out perplexity evaluation.
    /// </summary>
    public int EvaluationTokenStart { get; set; }

    /// <summary>
    /// Gets or sets an optional token count for the evaluation slice (<c>null</c> evaluates to the end of the corpus).
    /// </summary>
    public int? EvaluationTokenCount { get; set; }

    /// <summary>
    /// Gets or sets an optional calibration/training token range <c>(Start, Count)</c> that is asserted to have
    /// strictly zero overlap with the evaluation token slice before scoring.
    /// </summary>
    public (int Start, int Count)? CalibrationTokenRange { get; set; }

    /// <summary>
    /// Gets or sets the number of untimed warmup items executed prior to recording raw wall-clock latencies (default 1).
    /// </summary>
    public int WarmupItems { get; set; } = 1;
}
