using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>One immutable elite stored in a quality-diversity archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionArchiveEntry<TGenome>
{
    /// <summary>Initializes an archive entry.</summary>
    public EvolutionArchiveEntry(EvolutionCellKey cell, EvolutionCandidate<TGenome> candidate, EvolutionEvaluation evaluation)
    {
        Guard.NotNull(cell);
        Guard.NotNull(candidate);
        Guard.NotNull(evaluation);
        if (candidate.EvaluationId != evaluation.EvaluationId || candidate.CanonicalGenome.Id != evaluation.GenomeId)
            throw new ArgumentException("Candidate and evaluation identities must match.", nameof(evaluation));
        Cell = cell;
        Candidate = candidate;
        Evaluation = evaluation;
    }

    /// <summary>Gets the occupied cell.</summary>
    public EvolutionCellKey Cell { get; }
    /// <summary>Gets the canonical candidate.</summary>
    public EvolutionCandidate<TGenome> Candidate { get; }
    /// <summary>Gets the completed evaluation.</summary>
    public EvolutionEvaluation Evaluation { get; }
}
