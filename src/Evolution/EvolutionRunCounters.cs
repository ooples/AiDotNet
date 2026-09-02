using AiDotNet.Enums;

namespace AiDotNet.Evolution;

/// <summary>Immutable counters for all terminal evaluation statuses.</summary>
/// <remarks>
/// <para>
/// <see cref="Proposals"/> counts every identity the engine allocated, including seeds, duplicates, and candidates
/// that failed before evaluation. <see cref="EvaluationAttempts"/> counts calls into the task evaluator, so retries
/// add to it while cache hits and duplicates do not. <see cref="CompletedEvaluations"/> counts terminal
/// <see cref="EvolutionEvaluationStatus.Completed"/> outcomes, including those served from the evaluation cache.
/// <see cref="StatusCounts"/> holds one entry per terminal status that actually occurred; statuses that never
/// occurred are absent rather than zero. The dictionary is defensively copied at construction, so later changes to
/// the source do not leak into the snapshot.
/// </para>
/// <para><b>For Beginners:</b> This is the scoreboard you get at the end of a run through
/// <see cref="EvolutionRunResult{TGenome}"/>. It tells you how many candidates were suggested, how many actually ran
/// through your expensive evaluator, how many finished successfully, and how the rest ended - for example how many
/// were duplicates, timed out, or failed. If <see cref="EvaluationAttempts"/> is much lower than
/// <see cref="Proposals"/>, your variation operator is producing many duplicates; if the count for
/// <see cref="EvolutionEvaluationStatus.TimedOut"/> is high, the evaluation timeout is probably too tight.</para>
/// </remarks>
public sealed class EvolutionRunCounters
{
    private readonly IReadOnlyDictionary<EvolutionEvaluationStatus, long> _statusCounts;

    /// <summary>Initializes run counters.</summary>
    /// <param name="proposals">The non-negative number of allocated proposals.</param>
    /// <param name="evaluationAttempts">The non-negative number of evaluator calls, including retries.</param>
    /// <param name="completedEvaluations">The non-negative number of completed terminal evaluations.</param>
    /// <param name="statusCounts">Terminal counts by status; copied defensively.</param>
    /// <param name="abandonedEvaluations">
    /// The non-negative number of evaluator calls the engine stopped waiting for after the timeout grace period.
    /// </param>
    public EvolutionRunCounters(long proposals, long evaluationAttempts, long completedEvaluations,
        IReadOnlyDictionary<EvolutionEvaluationStatus, long> statusCounts, long abandonedEvaluations = 0)
    {
        if (proposals < 0) throw new ArgumentOutOfRangeException(nameof(proposals));
        if (evaluationAttempts < 0) throw new ArgumentOutOfRangeException(nameof(evaluationAttempts));
        if (completedEvaluations < 0) throw new ArgumentOutOfRangeException(nameof(completedEvaluations));
        if (abandonedEvaluations < 0) throw new ArgumentOutOfRangeException(nameof(abandonedEvaluations));
        if (statusCounts is null) throw new ArgumentNullException(nameof(statusCounts));
        Proposals = proposals;
        EvaluationAttempts = evaluationAttempts;
        CompletedEvaluations = completedEvaluations;
        AbandonedEvaluations = abandonedEvaluations;
        _statusCounts = new System.Collections.ObjectModel.ReadOnlyDictionary<EvolutionEvaluationStatus, long>(
            statusCounts.ToDictionary(item => item.Key, item => item.Value));
    }

    /// <summary>Gets all proposals, including duplicates and validation failures.</summary>
    public long Proposals { get; }
    /// <summary>Gets actual evaluator calls, including retries.</summary>
    public long EvaluationAttempts { get; }
    /// <summary>Gets completed terminal evaluations, including cache hits.</summary>
    public long CompletedEvaluations { get; }
    /// <summary>Gets terminal counts by status.</summary>
    public IReadOnlyDictionary<EvolutionEvaluationStatus, long> StatusCounts => _statusCounts;

    /// <summary>Gets evaluator calls the engine stopped waiting for after the timeout grace period elapsed.</summary>
    /// <remarks>
    /// Non-zero only when <c>EvolutionEngineOptions.EvaluationGracePeriod</c> is set. Each abandoned call was recorded
    /// as <see cref="EvolutionEvaluationStatus.TimedOut"/> and the batch continued without it; the call itself may keep
    /// running until it notices its cancellation token or finishes on its own. A persistently non-zero value means an
    /// evaluator is ignoring its cancellation token, which is worth fixing because abandoned work still consumes CPU.
    /// This counter depends on wall-clock timing and is therefore deliberately excluded from the run state hash.
    /// </remarks>
    public long AbandonedEvaluations { get; }
}
