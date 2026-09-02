using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Evolution;

/// <summary>The immutable result of an evolution run.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The constructor copies every island archive into an <see cref="EvolutionArchiveSnapshot{TGenome}"/>, so the
/// result is a stable snapshot that later archive activity cannot alter. <see cref="StateHash"/> is computed from
/// logical state only (counters, the deduplication set, the evaluation cache, adaptive-selection state, retained
/// failure diagnostics, and archive contents); elapsed times and observer behaviour are excluded, so two runs with
/// the same seed, options, and components must produce the same hash. That property is the engine's determinism
/// check and the basis of checkpoint compatibility testing.
/// </para>
/// <para><b>For Beginners:</b> This is what you get back when an evolution run finishes. <see cref="StopReason"/>
/// tells you why it ended (a budget ran out, the time limit was hit, a candidate failed under fail-fast, and so on),
/// <see cref="Counters"/> tells you how much work was done, and <see cref="Islands"/> holds the final archives, one
/// per island, each containing the best solution found for every behaviour cell. If you just want the single best
/// solution overall, read <see cref="Best"/>: it compares the top entry of every island by quality in the task's
/// optimisation direction and breaks ties by genome identity, so the answer is the same on every machine. Compare
/// <see cref="StateHash"/> across two runs to confirm they were identical, the way you would compare file
/// checksums.</para>
/// </remarks>
public sealed class EvolutionRunResult<TGenome>
{
    /// <summary>Initializes a run result.</summary>
    /// <param name="stopReason">Why the run stopped.</param>
    /// <param name="islands">The final island archives; each is snapshotted rather than referenced.</param>
    /// <param name="counters">Final run counters.</param>
    /// <param name="stateHash">The deterministic state hash computed by the engine.</param>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="islands"/>, <paramref name="counters"/>, or <paramref name="stateHash"/> is <c>null</c>.
    /// </exception>
    public EvolutionRunResult(EvolutionStopReason stopReason, IReadOnlyList<IEvolutionArchiveView<TGenome>> islands,
        EvolutionRunCounters counters, string stateHash)
    {
        StopReason = stopReason;
        if (islands is null) throw new ArgumentNullException(nameof(islands));
        Islands = Array.AsReadOnly(islands.Select(archive =>
            (IEvolutionArchiveView<TGenome>)new EvolutionArchiveSnapshot<TGenome>(archive)).ToArray());
        Counters = counters ?? throw new ArgumentNullException(nameof(counters));
        StateHash = stateHash ?? throw new ArgumentNullException(nameof(stateHash));
    }

    /// <summary>Gets why the run stopped.</summary>
    public EvolutionStopReason StopReason { get; }
    /// <summary>Gets the final island archives.</summary>
    public IReadOnlyList<IEvolutionArchiveView<TGenome>> Islands { get; }
    /// <summary>Gets final run counters.</summary>
    public EvolutionRunCounters Counters { get; }
    /// <summary>Gets a deterministic hash that excludes wall-clock timing and observer behavior.</summary>
    public string StateHash { get; }

    /// <summary>Gets the globally best elite using deterministic quality and identity tie-breaking.</summary>
    /// <remarks>
    /// Returns <c>null</c> when every island is empty. The comparison direction is read from the first island; the
    /// engine requires every island to share one archive definition, so all islands agree on it.
    /// </remarks>
    public EvolutionArchiveEntry<TGenome>? Best => Islands.Select(archive => archive.Best)
        .OfType<EvolutionArchiveEntry<TGenome>>()
        .OrderBy(entry => entry.Evaluation.Quality,
            Islands.Count == 0 || Islands[0].Direction == EvolutionOptimizationDirection.Maximize
                ? Comparer<double?>.Create((x, y) => Nullable.Compare(y, x))
                : Comparer<double?>.Default)
        .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal)
        .FirstOrDefault();
}
