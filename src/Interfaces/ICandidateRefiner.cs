using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Optionally improves a proposed genome while returning a new immutable snapshot.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The engine invokes the refiner once per proposal, after variation and before canonicalization, and supplies a
/// refiner-local <see cref="StableRandom"/> stream derived from the run seed and the proposal's evaluation ID. The
/// refiner must not mutate its input; when there is nothing to improve it may return the input unchanged, but it must
/// never return <see langword="null"/>. <see cref="Id"/> and <see cref="VersionHash"/> are folded into the engine's
/// compatibility hash, so changing the refinement algorithm invalidates old checkpoints instead of silently resuming a
/// run with different behavior.
/// </para>
/// <para><b>For Beginners:</b> Evolution proposes rough candidates by mutating and recombining parents, and a refiner
/// is an optional polishing step that runs on each candidate before it is scored. Think of it as a quick local tune-up:
/// if a genome encodes a neural-network layout, a refiner might run a few gradient steps so the evaluator sees that
/// layout at its best rather than at a random starting point. Use a refiner when a cheap inner optimizer makes the
/// expensive evaluation more informative; leave it out when the evaluation already trains or tunes the candidate
/// itself. Whatever the refiner does must depend only on its inputs and the supplied random stream, so that runs stay
/// reproducible and resumable from checkpoints.</para>
/// <para>
/// This is the memetic-algorithm pattern of pairing population search with per-individual local search (Moscato, 1989,
/// "On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms"). Refinement
/// runs before duplicate detection and cache lookup, so its cost is paid on every proposal; keep it small relative to
/// evaluation.
/// </para>
/// </remarks>
public interface ICandidateRefiner<TGenome>
{
    /// <summary>Gets a stable refiner identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Returns a refined genome without modifying the input object.</summary>
    /// <param name="genome">The proposed genome, which must be treated as read-only.</param>
    /// <param name="context">The evaluation identity and the refiner-local deterministic random stream.</param>
    /// <param name="cancellationToken">Cancellation propagated from the engine run.</param>
    /// <returns>A refined genome, or the unchanged input when no improvement applies; never <see langword="null"/>.</returns>
    ValueTask<TGenome> RefineAsync(TGenome genome, EvolutionRefinementContext context, CancellationToken cancellationToken = default);
}
