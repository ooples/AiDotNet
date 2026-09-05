using AiDotNet.Enums;
using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Mutable engine-owned quality-diversity archive contract.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A quality-diversity archive keeps the best evaluated candidate per behavior cell rather than a single global
/// winner (MAP-Elites: Mouret and Clune, "Illuminating search spaces by mapping elites", 2015, arXiv:1504.04909).
/// The engine owns the archive: it inserts completed evaluations through <see cref="TryAdd"/> and draws parents
/// through <see cref="Sample"/>. Read-only consumers such as observers and migration policies receive the
/// <see cref="IEvolutionArchiveView{TGenome}"/> base contract instead.
/// </para>
/// <para>
/// Implementations must be deterministic: identical insertion sequences must yield identical contents, ties must
/// be broken by stable ordinal identity rather than insertion order or hash codes, and all randomness must come
/// from the caller-supplied <see cref="StableRandom"/> so that archive state never depends on hidden generator
/// state. Only evaluations with status <see cref="EvolutionEvaluationStatus.Completed"/> and a present quality are
/// eligible for insertion; everything else must be reported as <see cref="EvolutionArchiveInsertionResult.Rejected"/>.
/// </para>
/// <para><b>For Beginners:</b> Picture a trophy shelf with one slot for each kind of solution, where "kind" is
/// defined by measurable behavior descriptors such as model family and configuration complexity. A new candidate
/// only takes a slot when it beats the current occupant of the same slot, so the shelf fills with the best simple
/// model, the best medium model, the best complex model, and so on, instead of dozens of near-copies of one winner.
/// <see cref="TryAdd"/> is how a finished evaluation competes for a slot, and <see cref="Sample"/> picks a random
/// occupied slot to use as the parent of the next proposal. Implement this interface when you need your own archive
/// layout; use <c>MapElitesArchive&lt;TGenome&gt;</c> when the standard grid archive is enough.</para>
/// </remarks>
public interface IEvolutionArchive<TGenome> : IEvolutionArchiveView<TGenome>
{
    /// <summary>Attempts to insert a completed candidate evaluation.</summary>
    /// <param name="candidate">The canonical candidate whose identity must match <paramref name="evaluation"/>.</param>
    /// <param name="evaluation">The completed evaluation supplying quality and descriptors.</param>
    /// <returns>Whether the candidate was inserted, replaced an incumbent, evicted another cell, did not improve, or was rejected.</returns>
    EvolutionArchiveInsertionResult TryAdd(EvolutionCandidate<TGenome> candidate, EvolutionEvaluation evaluation);

    /// <summary>Samples uniformly from occupied cells using a caller-owned stable stream.</summary>
    /// <param name="random">The deterministic stream that supplies the selection; the archive must not keep its own.</param>
    /// <returns>One occupied entry, or <c>null</c> when the archive is empty.</returns>
    EvolutionArchiveEntry<TGenome>? Sample(StableRandom random);
}
