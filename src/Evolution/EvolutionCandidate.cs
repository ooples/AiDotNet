using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>An immutable candidate assigned by the evolution engine.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionCandidate<TGenome>
{
    /// <summary>Initializes a candidate.</summary>
    public EvolutionCandidate(long evaluationId, EvolutionCanonicalGenome<TGenome> canonicalGenome, EvolutionLineage lineage)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        Guard.NotNull(canonicalGenome);
        Guard.NotNull(lineage);
        EvaluationId = evaluationId;
        CanonicalGenome = canonicalGenome;
        Lineage = lineage;
    }

    /// <summary>Gets the monotonically increasing evaluation identifier.</summary>
    public long EvaluationId { get; }

    /// <summary>Gets the canonical genome snapshot.</summary>
    public EvolutionCanonicalGenome<TGenome> CanonicalGenome { get; }

    /// <summary>Gets the immutable lineage.</summary>
    public EvolutionLineage Lineage { get; }
}
