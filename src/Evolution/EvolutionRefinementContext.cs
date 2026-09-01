namespace AiDotNet.Evolution;

/// <summary>Inputs available to an optional candidate refiner.</summary>
public sealed class EvolutionRefinementContext
{
    /// <summary>Initializes a refinement context.</summary>
    public EvolutionRefinementContext(long evaluationId, StableRandom random)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        EvaluationId = evaluationId;
        Random = random ?? throw new ArgumentNullException(nameof(random));
    }

    /// <summary>Gets the assigned evaluation ID.</summary>
    public long EvaluationId { get; }
    /// <summary>Gets a refiner-local stable random stream.</summary>
    public StableRandom Random { get; }
}
