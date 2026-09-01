namespace AiDotNet.Evolution;

/// <summary>Immutable evaluation cost and elapsed-time metadata.</summary>
public sealed class EvolutionEvaluationCost
{
    /// <summary>Initializes cost metadata.</summary>
    public EvolutionEvaluationCost(TimeSpan elapsed, int attemptCount, double costUnits)
    {
        if (elapsed < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(elapsed));
        if (attemptCount < 0) throw new ArgumentOutOfRangeException(nameof(attemptCount));
        if (!EvolutionDescriptorDefinition.IsFinite(costUnits) || costUnits < 0) throw new ArgumentOutOfRangeException(nameof(costUnits));
        Elapsed = elapsed;
        AttemptCount = attemptCount;
        CostUnits = costUnits;
    }

    /// <summary>Gets wall-clock evaluator time.</summary>
    public TimeSpan Elapsed { get; }

    /// <summary>Gets the one-based attempt count for this canonical candidate.</summary>
    public int AttemptCount { get; }

    /// <summary>Gets task-defined resource units.</summary>
    public double CostUnits { get; }
}
