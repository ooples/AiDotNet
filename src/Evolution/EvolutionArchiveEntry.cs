using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>One immutable elite stored in a quality-diversity archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// An entry binds together the <see cref="Cell"/> a candidate landed in, the canonical <see cref="Candidate"/> itself,
/// and the completed <see cref="Evaluation"/> that placed it there. The constructor verifies that the candidate and
/// evaluation share the same evaluation ID and canonical genome ID, so an entry can never pair a genome with another
/// candidate's score. Entries are immutable: islands share them by reference during migration and checkpoints copy them
/// verbatim.
/// </para>
/// <para><b>For Beginners:</b> A MAP-Elites archive is a grid of cells, one for each combination of behavior
/// characteristics, and each occupied cell holds the single best candidate seen with that behavior. This class is what
/// sits in a cell: the candidate, its score and descriptors, and the cell coordinates. When you inspect a finished run
/// you iterate the archive's entries, read <c>Evaluation.Quality</c> to see how good each elite is, and pull the actual
/// solution out of <c>Candidate.CanonicalGenome.Genome</c>. For example, the entry in the "small, shallow" cell of an
/// architecture search is the most accurate small, shallow network the run discovered.</para>
/// </remarks>
public sealed class EvolutionArchiveEntry<TGenome>
{
    /// <summary>Initializes an archive entry.</summary>
    /// <param name="cell">The cell occupied by the candidate.</param>
    /// <param name="candidate">The canonical candidate.</param>
    /// <param name="evaluation">The completed evaluation of <paramref name="candidate"/>.</param>
    /// <exception cref="ArgumentException">The candidate and evaluation identities do not match.</exception>
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
