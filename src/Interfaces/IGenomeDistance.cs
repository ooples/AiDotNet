namespace AiDotNet.Interfaces;

/// <summary>Measures structural distance between two genomes without evaluating either of them.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The evolution engine uses this contract for its structural novelty gate: before a proposal is handed to the
/// evaluator, the engine compares it against the elites of its target island and rejects it when the smallest
/// distance falls below <c>EvolutionEngineOptions.NoveltyDistanceThreshold</c>. One rejection therefore costs at
/// most one distance call per occupied cell of that island, with no evaluator call, no embedding request, and no
/// network round trip at all, which is the whole point of expressing near-duplicate detection as a distance rather
/// than as an embedding similarity or a language-model judgement.
/// </para>
/// <para>
/// Implementations must be pure, symmetric, and deterministic: <c>Distance(a, b)</c> must equal <c>Distance(b, a)</c>,
/// must return the same finite non-negative value on every call and every target framework, and must return zero for
/// two genomes the task would canonicalize to the same identity. <see cref="Id"/> and <see cref="VersionHash"/> are
/// folded into the engine's checkpoint compatibility hash, so change <see cref="VersionHash"/> whenever the metric
/// changes; otherwise a resumed run would silently apply a different novelty rule than the run that wrote the
/// checkpoint.
/// </para>
/// <para><b>For Beginners:</b> Evolution wastes evaluator time when it keeps proposing candidates that are almost
/// the same as ones already tried. Exact duplicate detection catches only byte-identical repeats, so this interface
/// lets you say how different two candidates really are, returning a small number for near-twins and a larger one
/// for genuinely new ideas. For a numeric genome that could be the absolute difference between two values; for a
/// program genome it could be a token-overlap score. Supply one to the engine together with a threshold and the
/// engine will discard near-duplicates before they ever reach your expensive scoring function.</para>
/// </remarks>
public interface IGenomeDistance<TGenome>
{
    /// <summary>Gets a stable metric identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Computes the finite, non-negative, symmetric distance between two genomes.</summary>
    /// <param name="first">The first genome.</param>
    /// <param name="second">The second genome.</param>
    /// <returns>Zero for structurally identical genomes and a larger value the more they differ.</returns>
    double Distance(TGenome first, TGenome second);
}
