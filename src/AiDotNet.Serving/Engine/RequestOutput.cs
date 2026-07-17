namespace AiDotNet.Serving.Engine;

/// <summary>
/// One output stream (one <see cref="Sequence"/>) of a request as reported by the engine: the tokens produced
/// so far and, once terminal, why it stopped. A request with <c>N &gt; 1</c> yields several of these, one per
/// <see cref="SequenceIndex"/>.
/// </summary>
public sealed class CompletionOutput
{
    /// <summary>Creates a completion output snapshot.</summary>
    public CompletionOutput(int sequenceIndex, IReadOnlyList<int> tokenIds, bool isFinished, string? finishReason)
    {
        SequenceIndex = sequenceIndex;
        TokenIds = tokenIds ?? throw new ArgumentNullException(nameof(tokenIds));
        IsFinished = isFinished;
        FinishReason = finishReason;
    }

    /// <summary>0-based index of this output stream within its request.</summary>
    public int SequenceIndex { get; }

    /// <summary>Generated token ids for this stream (excludes the prompt).</summary>
    public IReadOnlyList<int> TokenIds { get; }

    /// <summary>Number of generated tokens.</summary>
    public int Count => TokenIds.Count;

    /// <summary>True once this stream reached a terminal state.</summary>
    public bool IsFinished { get; }

    /// <summary>Reason it stopped ("stop", "length", "abort"), or null while still running.</summary>
    public string? FinishReason { get; }
}

/// <summary>
/// The engine's per-step report for a request: its prompt, every output stream's progress, and whether the
/// whole request is done. In streaming mode the engine emits one of these per step carrying newly appended
/// tokens; in non-streaming mode a single terminal output carries the full result.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> this is the engine's answer envelope for one request. Each engine step, the
/// engine hands back a RequestOutput saying "here is what request X looks like now" — which tokens exist and
/// whether it has finished. The front end turns these into the streamed chunks a client sees.</para>
/// </remarks>
public sealed class RequestOutput
{
    /// <summary>Creates a request output snapshot.</summary>
    public RequestOutput(
        string requestId,
        IReadOnlyList<int> promptTokenIds,
        IReadOnlyList<CompletionOutput> outputs,
        bool isFinished)
    {
        RequestId = requestId ?? throw new ArgumentNullException(nameof(requestId));
        PromptTokenIds = promptTokenIds ?? throw new ArgumentNullException(nameof(promptTokenIds));
        Outputs = outputs ?? throw new ArgumentNullException(nameof(outputs));
        IsFinished = isFinished;
    }

    /// <summary>Id of the request this output belongs to.</summary>
    public string RequestId { get; }

    /// <summary>The request's prompt token ids (for echo / logprob alignment).</summary>
    public IReadOnlyList<int> PromptTokenIds { get; }

    /// <summary>Per-stream outputs (one per sampled sequence).</summary>
    public IReadOnlyList<CompletionOutput> Outputs { get; }

    /// <summary>True once every output stream of the request has finished.</summary>
    public bool IsFinished { get; }
}
