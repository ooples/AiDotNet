using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>An immutable candidate assigned by the evolution engine.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A candidate binds three things the engine fixes before evaluation begins: a monotonically increasing
/// <see cref="EvaluationId"/> that orders every proposal in the run, the
/// <see cref="EvolutionCanonicalGenome{TGenome}"/> snapshot whose identifier drives duplicate detection and
/// evaluation caching, and the <see cref="EvolutionLineage"/> recording parents, inspirations, operator
/// identifiers, generation, and island. The same instance flows through
/// <see cref="EvolutionEventKind.Proposed"/> and <see cref="EvolutionEventKind.Evaluated"/> events and into
/// <see cref="EvolutionArchiveEntry{TGenome}"/>, so consumers can correlate by reference or by
/// <see cref="EvaluationId"/>.
/// </para>
/// <para><b>For Beginners:</b> Think of a candidate as one entry ticket in the evolution run: it has a number (the
/// evaluation ID), the exact solution being tested (the canonical genome), and a family tree saying where the idea
/// came from (the lineage). Because it is immutable, the same ticket can be reported to observers, stored in the
/// archive, and written to a checkpoint without anyone accidentally changing it. You receive candidates in observer
/// events and in your task's evaluate method; you rarely construct them yourself.</para>
/// </remarks>
public sealed class EvolutionCandidate<TGenome>
{
    /// <summary>Initializes a candidate.</summary>
    /// <param name="evaluationId">The non-negative evaluation identifier assigned by the engine.</param>
    /// <param name="canonicalGenome">The canonical genome snapshot.</param>
    /// <param name="lineage">The immutable lineage record.</param>
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
