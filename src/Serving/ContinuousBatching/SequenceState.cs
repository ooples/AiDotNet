namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// Represents the state of a single sequence being processed in continuous batching.
/// </summary>
/// <remarks>
/// <para>
/// Each sequence tracks its own progress through generation, including tokens generated,
/// KV-cache state, and completion status. This enables sequences to be added to and
/// removed from batches dynamically.
/// </para>
/// <para><b>For Beginners:</b> Think of this as tracking one person's order in a restaurant.
///
/// Traditional batching: Everyone orders at once, waits together, gets food together.
/// Continuous batching: People can order anytime, food comes when ready, new orders join ongoing batch.
///
/// SequenceState tracks:
/// - What tokens have been generated so far
/// - When this request started
/// - Whether generation is complete
/// - How many tokens are left to generate
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
public class SequenceState<T>
{
    private static long _nextId = 0;

    /// <summary>
    /// Unique identifier for this sequence.
    /// </summary>
    public long SequenceId { get; }

    /// <summary>
    /// The original request that created this sequence.
    /// </summary>
    public GenerationRequest<T> Request { get; }

    /// <summary>
    /// Current status of this sequence.
    /// </summary>
    public SequenceStatus Status { get; set; }

    /// <summary>
    /// List of token IDs generated so far (including prompt tokens).
    /// </summary>
    public List<int> TokenIds { get; }

    /// <summary>
    /// Number of tokens from the original prompt.
    /// </summary>
    public int PromptLength { get; }

    /// <summary>
    /// Number of tokens generated (excluding prompt).
    /// </summary>
    public int GeneratedLength => TokenIds.Count - PromptLength;

    /// <summary>
    /// Maximum number of new tokens to generate.
    /// </summary>
    public int MaxNewTokens { get; }

    /// <summary>
    /// Timestamp when the sequence was created.
    /// </summary>
    public DateTime CreatedAt { get; }

    /// <summary>
    /// Timestamp when generation started (after prefill).
    /// </summary>
    public DateTime? GenerationStartedAt { get; set; }

    /// <summary>
    /// Timestamp when generation completed.
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Index in the current batch (-1 if not in batch).
    /// </summary>
    public int BatchIndex { get; set; } = -1;

    /// <summary>
    /// Cache slot index for this sequence.
    /// </summary>
    public int CacheSlot { get; set; } = -1;

    /// <summary>
    /// Whether the prefill phase is complete.
    /// </summary>
    public bool PrefillComplete { get; set; } = false;

    /// <summary>
    /// Stop reason if generation is complete.
    /// </summary>
    public StopReason? FinishReason { get; set; }

    /// <summary>
    /// Cumulative log probability of generated tokens.
    /// </summary>
    public double CumulativeLogProb { get; set; } = 0.0;

    /// <summary>
    /// Priority for scheduling (higher = more important).
    /// </summary>
    public int Priority { get; set; } = 0;

    /// <summary>
    /// Optional user context associated with this sequence.
    /// </summary>
    public object? UserContext { get; set; }

    /// <summary>
    /// Creates a new sequence state from a generation request.
    /// </summary>
    public SequenceState(GenerationRequest<T> request)
    {
        SequenceId = Interlocked.Increment(ref _nextId);
        Guard.NotNull(request);
        Request = request;
        Status = SequenceStatus.Pending;
        TokenIds = new List<int>(request.PromptTokenIds);
        PromptLength = request.PromptTokenIds.Count;
        MaxNewTokens = request.MaxNewTokens;
        CreatedAt = DateTime.UtcNow;
        Priority = request.Priority;
        UserContext = request.UserContext;
    }

    /// <summary>
    /// Appends a newly generated token to the sequence.
    /// </summary>
    /// <param name="tokenId">The generated token ID.</param>
    /// <param name="logProb">Log probability of the token.</param>
    public void AppendToken(int tokenId, double logProb = 0.0)
    {
        TokenIds.Add(tokenId);
        CumulativeLogProb += logProb;
    }

    /// <summary>
    /// Checks if generation should stop based on various conditions.
    /// </summary>
    /// <param name="eosTokenId">End-of-sequence token ID.</param>
    /// <param name="stopTokenIds">Additional stop token IDs.</param>
    /// <returns>True if generation should stop.</returns>
    public bool ShouldStop(int eosTokenId, IReadOnlyCollection<int>? stopTokenIds = null)
    {
        // Check max length
        if (GeneratedLength >= MaxNewTokens)
        {
            FinishReason = StopReason.MaxLength;
            return true;
        }

        // Check for EOS token
        if (TokenIds.Count > 0 && TokenIds[^1] == eosTokenId)
        {
            FinishReason = StopReason.EndOfSequence;
            return true;
        }

        // Check for stop tokens
        if (stopTokenIds != null && TokenIds.Count > 0 && stopTokenIds.Contains(TokenIds[^1]))
        {
            FinishReason = StopReason.StopToken;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Gets the time spent in queue (before generation started).
    /// </summary>
    public TimeSpan QueueTime => (GenerationStartedAt ?? DateTime.UtcNow) - CreatedAt;

    /// <summary>
    /// Gets the total generation time (after prefill).
    /// </summary>
    public TimeSpan? GenerationTime => GenerationStartedAt.HasValue
        ? (CompletedAt ?? DateTime.UtcNow) - GenerationStartedAt.Value
        : null;

    /// <summary>
    /// Gets tokens per second for this sequence.
    /// </summary>
    public double? TokensPerSecond => GenerationTime.HasValue && GenerationTime.Value.TotalSeconds > 0
        ? GeneratedLength / GenerationTime.Value.TotalSeconds
        : null;

    /// <summary>
    /// Marks the sequence as complete.
    /// </summary>
    public void Complete(StopReason reason)
    {
        Status = SequenceStatus.Completed;
        FinishReason = reason;
        CompletedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Marks the sequence as cancelled.
    /// </summary>
    public void Cancel()
    {
        Status = SequenceStatus.Cancelled;
        FinishReason = StopReason.Cancelled;
        CompletedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Marks the sequence as failed.
    /// </summary>
    public void Fail(string? errorMessage = null)
    {
        Status = SequenceStatus.Failed;
        FinishReason = StopReason.Error;
        CompletedAt = DateTime.UtcNow;
    }
}

/// <summary>
/// Status of a sequence in the continuous batching system.
/// </summary>
public enum SequenceStatus
{
    /// <summary>Sequence is waiting to be processed.</summary>
    Pending,

    /// <summary>Sequence is being prefilled (processing prompt).</summary>
    Prefilling,

    /// <summary>Sequence is actively generating tokens.</summary>
    Generating,

    /// <summary>Sequence has completed generation.</summary>
    Completed,

    /// <summary>Sequence was cancelled.</summary>
    Cancelled,

    /// <summary>Sequence encountered an error.</summary>
    Failed,

    /// <summary>Sequence is paused (preempted for higher priority).</summary>
    Paused
}

/// <summary>
/// Reasons why generation stopped.
/// </summary>
public enum StopReason
{
    /// <summary>Reached maximum token limit.</summary>
    MaxLength,

    /// <summary>Generated end-of-sequence token.</summary>
    EndOfSequence,

    /// <summary>Generated a stop token.</summary>
    StopToken,

    /// <summary>Request was cancelled by user.</summary>
    Cancelled,

    /// <summary>An error occurred during generation.</summary>
    Error
}

/// <summary>
/// Represents a request for text generation.
/// </summary>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
public class GenerationRequest<T>
{
    /// <summary>
    /// Token IDs of the prompt.
    /// </summary>
    public List<int> PromptTokenIds { get; set; } = new();

    /// <summary>
    /// Maximum number of new tokens to generate.
    /// </summary>
    public int MaxNewTokens { get; set; } = 100;

    /// <summary>
    /// Temperature for sampling (higher = more random).
    /// </summary>
    public float Temperature { get; set; } = 1.0f;

    /// <summary>
    /// Top-p (nucleus) sampling threshold.
    /// </summary>
    public float TopP { get; set; } = 1.0f;

    /// <summary>
    /// Top-k sampling (0 = disabled).
    /// </summary>
    public int TopK { get; set; } = 0;

    /// <summary>
    /// Repetition penalty (1.0 = no penalty).
    /// </summary>
    public float RepetitionPenalty { get; set; } = 1.0f;

    /// <summary>
    /// Whether to use beam search.
    /// </summary>
    public bool UseBeamSearch { get; set; } = false;

    /// <summary>
    /// Number of beams for beam search.
    /// </summary>
    public int NumBeams { get; set; } = 1;

    /// <summary>
    /// Priority for scheduling (higher = more important).
    /// </summary>
    public int Priority { get; set; } = 0;

    /// <summary>
    /// Optional user context.
    /// </summary>
    public object? UserContext { get; set; }

    /// <summary>
    /// Callback for streaming tokens.
    /// </summary>
    public Action<int>? OnTokenGenerated { get; set; }

    /// <summary>
    /// Additional stop token IDs.
    /// </summary>
    public List<int>? StopTokenIds { get; set; }
}
