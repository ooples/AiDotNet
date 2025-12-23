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

    /// <summary>Number of tokens generated.</summary>
    public int GeneratedLength { get; set; }

    /// <summary>Time spent waiting in queue.</summary>
    public TimeSpan QueueTime { get; set; }

    /// <summary>Time spent generating.</summary>
    public TimeSpan? GenerationTime { get; set; }

    /// <summary>Generation speed.</summary>
    public double? TokensPerSecond { get; set; }
}
