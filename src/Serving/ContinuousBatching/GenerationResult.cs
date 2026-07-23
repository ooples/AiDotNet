namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// Result of a generation request.
/// </summary>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
public class GenerationResult<T>
{
    /// <summary>Unique ID of the sequence.</summary>
    public long SequenceId { get; set; }

    /// <summary>All token IDs including prompt.</summary>
    public List<int> TokenIds { get; set; } = new List<int>();

    /// <summary>Only the generated tokens (excluding prompt).</summary>
    public List<int> GeneratedTokens { get; set; } = new List<int>();

    /// <summary>Reason why generation stopped.</summary>
    public StopReason FinishReason { get; set; }

    /// <summary>
    /// A human-readable error message when <see cref="FinishReason"/> is <see cref="StopReason.Error"/>
    /// (e.g. a structured-output constraint reached an unsatisfiable dead-end); null otherwise.
    /// </summary>
    public string? Error { get; set; }

    /// <summary>Number of tokens generated.</summary>
    public int GeneratedLength { get; set; }

    /// <summary>Time spent waiting in queue.</summary>
    public TimeSpan QueueTime { get; set; }

    /// <summary>Time spent generating.</summary>
    public TimeSpan? GenerationTime { get; set; }

    /// <summary>Generation speed.</summary>
    public double? TokensPerSecond { get; set; }

    /// <summary>
    /// Per-generated-token log-probability records, present only when the request set
    /// <see cref="GenerationRequest{T}.IncludeLogProbs"/>. One entry per generated token, in order.
    /// </summary>
    public List<PositionLogProbs>? LogProbs { get; set; }
}
