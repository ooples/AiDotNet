using AiDotNet.Enums;

namespace AiDotNet.Evolution;

/// <summary>One structured observer notification.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The engine reports progress to an <see cref="AiDotNet.Interfaces.IEvolutionObserver{TGenome}"/> as a stream of
/// these immutable events instead of writing to a console or logger. <see cref="Kind"/> says what happened,
/// <see cref="Sequence"/> increases monotonically within a run so consumers can order buffered events, and the
/// optional payload properties are populated only when they apply: <see cref="Candidate"/> accompanies
/// <see cref="EvolutionEventKind.Proposed"/>, <see cref="EvolutionEventKind.Evaluated"/>, and
/// <see cref="EvolutionEventKind.ArchiveChanged"/> events, <see cref="Evaluation"/> and <see cref="InsertionResult"/>
/// accompany <see cref="EvolutionEventKind.Evaluated"/> and <see cref="EvolutionEventKind.ArchiveChanged"/> events,
/// and <see cref="Message"/> carries a short note for
/// <see cref="EvolutionEventKind.Migrated"/>, <see cref="EvolutionEventKind.Checkpointed"/>, and
/// <see cref="EvolutionEventKind.Stopped"/> events, such as the stop reason.
/// </para>
/// <para>
/// Events are delivered on the engine's own execution path, so an observer should return quickly and must not call
/// back into the engine; anything expensive belongs on a queue drained elsewhere. The deterministic state hash of a
/// run excludes observer behavior, so a slow or failing observer cannot change search results.
/// </para>
/// <para><b>For Beginners:</b> If the evolution engine is a factory, events are the entries it writes in its log
/// book: "candidate 17 was proposed", "candidate 17 scored 0.83", "candidate 17 replaced the previous best in cell
/// 2,4". Write a small class that implements <see cref="AiDotNet.Interfaces.IEvolutionObserver{TGenome}"/>, hand it
/// to the engine, and you receive every entry as it happens. A typical use is a progress display that counts how
/// many <c>ArchiveChanged</c> events carried an insertion result of <c>Inserted</c> or <c>Replaced</c>, which shows
/// whether the search is still discovering improvements or has plateaued.</para>
/// </remarks>
public sealed class EvolutionEvent<TGenome>
{
    /// <summary>Initializes an observer event.</summary>
    /// <param name="kind">The event classification.</param>
    /// <param name="sequence">The nonnegative, monotonically increasing sequence number within the run.</param>
    /// <param name="candidate">The related candidate, when the event concerns one.</param>
    /// <param name="evaluation">The related terminal evaluation, when one exists.</param>
    /// <param name="insertionResult">The archive insertion result, for evaluation and archive events.</param>
    /// <param name="message">An optional short human-readable note.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="sequence"/> is negative.</exception>
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
