namespace AiDotNet.Serving.Engine;

/// <summary>
/// The engine's mutable, per-output-stream generation state for a request: the running token list (prompt +
/// generated), the lifecycle <see cref="SequenceState"/>, and the bookkeeping the scheduler and KV-cache
/// manager need. A request with <c>N &gt; 1</c> produces N sequences that share the prompt prefix.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> a Sequence is "one answer being written, one token at a time." It starts as the
/// prompt tokens; each engine step may append one generated token. The engine tracks how far it has been
/// processed (prefill vs decode), whether it is finished, and which KV-cache blocks hold its attention state.
/// This class holds the logical state only — the physical KV blocks live in the KV-cache manager, keyed by
/// <see cref="SequenceId"/>.</para>
/// </remarks>
public sealed class Sequence
{
    private readonly List<int> _tokenIds;

    /// <summary>Creates a sequence from a request and its per-sequence index (0..N-1).</summary>
    public Sequence(GenerationRequest request, int sequenceIndex)
    {
        Request = request ?? throw new ArgumentNullException(nameof(request));
        SequenceIndex = sequenceIndex;
        SequenceId = request.SamplingParameters.N > 1 ? $"{request.RequestId}#{sequenceIndex}" : request.RequestId;
        _tokenIds = new List<int>(request.PromptTokenIds);
        PromptLength = request.PromptLength;
        State = SequenceState.Waiting;
        NumComputedTokens = 0;
    }

    /// <summary>The originating request (prompt, sampling params).</summary>
    public GenerationRequest Request { get; }

    /// <summary>0-based index of this output stream within its request (for parallel sampling, N &gt; 1).</summary>
    public int SequenceIndex { get; }

    /// <summary>Unique id for this sequence's KV state (== RequestId when N == 1).</summary>
    public string SequenceId { get; }

    /// <summary>Number of prompt tokens (the prefill length).</summary>
    public int PromptLength { get; }

    /// <summary>Lifecycle state.</summary>
    public SequenceState State { get; set; }

    /// <summary>Reason a finished sequence stopped ("stop", "length", "abort"), or null while running.</summary>
    public string? FinishReason { get; private set; }

    /// <summary>All token ids so far: prompt followed by generated tokens.</summary>
    public IReadOnlyList<int> TokenIds => _tokenIds;

    /// <summary>Total tokens (prompt + generated).</summary>
    public int Length => _tokenIds.Count;

    /// <summary>Number of tokens generated so far (excludes the prompt).</summary>
    public int GeneratedLength => _tokenIds.Count - PromptLength;

    /// <summary>
    /// Number of tokens whose KV has already been computed and cached. During prefill this advances from 0 to
    /// <see cref="PromptLength"/> (possibly in chunks — chunked prefill); during decode it tracks Length - 1.
    /// The number of tokens still needing computation this step is <see cref="Length"/> - this value.
    /// </summary>
    public int NumComputedTokens { get; set; }

    /// <summary>True once prefill is complete and the sequence is in the decode phase (one token/step).</summary>
    public bool IsInDecodePhase => NumComputedTokens >= PromptLength;

    /// <summary>Appends a newly generated token id.</summary>
    public void AppendToken(int tokenId) => _tokenIds.Add(tokenId);

    /// <summary>Marks the sequence finished with the given terminal state and reason.</summary>
    public void Finish(SequenceState terminalState, string reason)
    {
        if (!terminalState.IsFinished())
            throw new ArgumentException("Finish requires a terminal state.", nameof(terminalState));
        State = terminalState;
        FinishReason = reason;
    }
}
