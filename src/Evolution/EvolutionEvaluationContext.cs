namespace AiDotNet.Evolution;

/// <summary>Deterministic evaluator context for one candidate.</summary>
public sealed class EvolutionEvaluationContext
{
    /// <summary>Initializes an evaluator context.</summary>
    public EvolutionEvaluationContext(long evaluationId, ulong rootSeed, ulong seedStream, int attemptCount)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        if (attemptCount <= 0) throw new ArgumentOutOfRangeException(nameof(attemptCount));
        EvaluationId = evaluationId;
        RootSeed = rootSeed;
        SeedStream = seedStream;
        AttemptCount = attemptCount;
    }

    /// <summary>Gets the evaluation ID.</summary>
    public long EvaluationId { get; }
    /// <summary>Gets the run root seed.</summary>
    public ulong RootSeed { get; }
    /// <summary>Gets the stable stream identifier.</summary>
    public ulong SeedStream { get; }
    /// <summary>Gets the one-based attempt count.</summary>
    public int AttemptCount { get; }

    /// <summary>Creates a fresh task-local stream whose sequence is independent of worker scheduling.</summary>
    public StableRandom CreateRandom() => StableRandom.CreateStream(RootSeed, SeedStream);
}
