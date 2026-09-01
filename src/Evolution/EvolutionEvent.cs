using AiDotNet.Enums;

namespace AiDotNet.Evolution;

/// <summary>One structured observer notification.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionEvent<TGenome>
{
    /// <summary>Initializes an observer event.</summary>
    public EvolutionEvent(EvolutionEventKind kind, long sequence, EvolutionCandidate<TGenome>? candidate = null,
        EvolutionEvaluation? evaluation = null, EvolutionArchiveInsertionResult? insertionResult = null, string? message = null)
    {
        if (sequence < 0) throw new ArgumentOutOfRangeException(nameof(sequence));
        Kind = kind;
        Sequence = sequence;
        Candidate = candidate;
        Evaluation = evaluation;
        InsertionResult = insertionResult;
        Message = message;
    }

    /// <summary>Gets the event kind.</summary>
    public EvolutionEventKind Kind { get; }
    /// <summary>Gets the monotonically increasing event sequence.</summary>
    public long Sequence { get; }
    /// <summary>Gets the related candidate, when applicable.</summary>
    public EvolutionCandidate<TGenome>? Candidate { get; }
    /// <summary>Gets the related evaluation, when applicable.</summary>
    public EvolutionEvaluation? Evaluation { get; }
    /// <summary>Gets the archive insertion result, when applicable.</summary>
    public EvolutionArchiveInsertionResult? InsertionResult { get; }
    /// <summary>Gets an optional bounded message.</summary>
    public string? Message { get; }
}
