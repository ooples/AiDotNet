using AiDotNet.Enums;

namespace AiDotNet.Evolution;

/// <summary>Immutable counters for all terminal evaluation statuses.</summary>
public sealed class EvolutionRunCounters
{
    private readonly IReadOnlyDictionary<EvolutionEvaluationStatus, long> _statusCounts;

    /// <summary>Initializes run counters.</summary>
    public EvolutionRunCounters(long proposals, long evaluationAttempts, long completedEvaluations,
        IReadOnlyDictionary<EvolutionEvaluationStatus, long> statusCounts)
    {
        if (proposals < 0) throw new ArgumentOutOfRangeException(nameof(proposals));
        if (evaluationAttempts < 0) throw new ArgumentOutOfRangeException(nameof(evaluationAttempts));
        if (completedEvaluations < 0) throw new ArgumentOutOfRangeException(nameof(completedEvaluations));
        Proposals = proposals;
        EvaluationAttempts = evaluationAttempts;
        CompletedEvaluations = completedEvaluations;
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
}
