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
    /// <param name="globalElites">
    /// The cross-island elite index in best-first order, or <c>null</c> when the index is disabled.
    /// </param>
    /// <param name="islandStatuses">
    /// One status snapshot per island in island order, or <c>null</c> when statuses were not captured.
    /// </param>
    /// <param name="retainedFailures">
    /// The bounded failure diagnostics retained by the engine, oldest first, or <c>null</c> when none were retained.
    /// </param>
    /// <param name="pendingArtifacts">
    /// Evaluator artifacts still queued for delivery to a future proposal, keyed by canonical genome identifier, or
    /// <c>null</c> when none are queued.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="islands"/>, <paramref name="counters"/>, or <paramref name="stateHash"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="globalElites"/> or <paramref name="islandStatuses"/> contains a <c>null</c> element, the status
    /// count does not match the island count, or <paramref name="pendingArtifacts"/> contains a <c>null</c> list.
    /// </exception>
    public EvolutionRunResult(EvolutionStopReason stopReason, IReadOnlyList<IEvolutionArchiveView<TGenome>> islands,
        EvolutionRunCounters counters, string stateHash,
        IReadOnlyList<EvolutionEliteRecord<TGenome>>? globalElites = null,
        IReadOnlyList<EvolutionIslandStatus>? islandStatuses = null,
        IReadOnlyList<EvolutionDiagnostic>? retainedFailures = null,
        IReadOnlyDictionary<string, IReadOnlyList<EvolutionArtifact>>? pendingArtifacts = null)
    {
        StopReason = stopReason;
        if (islands is null) throw new ArgumentNullException(nameof(islands));
        Islands = Array.AsReadOnly(islands.Select(archive =>
            (IEvolutionArchiveView<TGenome>)new EvolutionArchiveSnapshot<TGenome>(archive)).ToArray());
        Counters = counters ?? throw new ArgumentNullException(nameof(counters));
        StateHash = stateHash ?? throw new ArgumentNullException(nameof(stateHash));
        EvolutionEliteRecord<TGenome>[] elites = globalElites?.ToArray() ?? Array.Empty<EvolutionEliteRecord<TGenome>>();
        if (elites.Any(record => record is null))
            throw new ArgumentException("The global elite index cannot contain null records.", nameof(globalElites));
        GlobalElites = Array.AsReadOnly(elites);
        EvolutionIslandStatus[] statuses = islandStatuses?.ToArray() ?? Array.Empty<EvolutionIslandStatus>();
        if (statuses.Any(status => status is null))
            throw new ArgumentException("Island statuses cannot contain null entries.", nameof(islandStatuses));
        if (statuses.Length != 0 && statuses.Length != Islands.Count)
            throw new ArgumentException("Island statuses must cover every island.", nameof(islandStatuses));
        IslandStatuses = Array.AsReadOnly(statuses);
        EvolutionDiagnostic[] failures = retainedFailures?.ToArray() ?? Array.Empty<EvolutionDiagnostic>();
        if (failures.Any(diagnostic => diagnostic is null))
            throw new ArgumentException("Retained failures cannot contain null diagnostics.", nameof(retainedFailures));
        RetainedFailures = Array.AsReadOnly(failures);
        var pending = new Dictionary<string, IReadOnlyList<EvolutionArtifact>>(StringComparer.Ordinal);
        if (pendingArtifacts is not null)
        {
            foreach (KeyValuePair<string, IReadOnlyList<EvolutionArtifact>> entry in pendingArtifacts)
            {
                if (string.IsNullOrWhiteSpace(entry.Key) || entry.Value is null)
                    throw new ArgumentException("Pending artifacts require a genome identifier and a list.", nameof(pendingArtifacts));
                pending[entry.Key] = Array.AsReadOnly(entry.Value.ToArray());
            }
        }
        PendingArtifacts = new System.Collections.ObjectModel.ReadOnlyDictionary<string, IReadOnlyList<EvolutionArtifact>>(pending);
    }

    /// <summary>Gets why the run stopped.</summary>
    public EvolutionStopReason StopReason { get; }
    /// <summary>Gets the final island archives.</summary>
    public IReadOnlyList<IEvolutionArchiveView<TGenome>> Islands { get; }
    /// <summary>Gets final run counters.</summary>
    public EvolutionRunCounters Counters { get; }
    /// <summary>Gets a deterministic hash that excludes wall-clock timing and observer behavior.</summary>
    public string StateHash { get; }

    /// <summary>Gets the cross-island global elites in best-first order; empty when the index is disabled.</summary>
    /// <remarks>
    /// Populated when <c>EvolutionEngineOptions.GlobalEliteCount</c> is positive. Unlike <see cref="Best"/>, which
    /// reports only the single leading elite, this list spans every island and is bounded by the configured count.
    /// </remarks>
    public IReadOnlyList<EvolutionEliteRecord<TGenome>> GlobalElites { get; }

    /// <summary>Gets one progress snapshot per island, in island order; empty when statuses were not captured.</summary>
    public IReadOnlyList<EvolutionIslandStatus> IslandStatuses { get; }

    /// <summary>Gets the retained failure diagnostics, oldest first and bounded by <c>MaxRetainedFailures</c>.</summary>
    /// <remarks>
    /// Includes evaluator failures and the <c>descriptor_missing:&lt;name&gt;</c> diagnostic the engine raises when a
    /// completed evaluation omits a configured archive descriptor and would otherwise be rejected silently.
    /// </remarks>
    public IReadOnlyList<EvolutionDiagnostic> RetainedFailures { get; }

    /// <summary>Gets evaluator artifacts still queued for a future proposal, keyed by canonical genome identifier.</summary>
    /// <remarks>
    /// Empty unless <c>EvolutionEngineOptions.Artifacts.Enabled</c> and
    /// <c>EvolutionEngineOptions.Artifacts.DeliverToNextProposal</c> are both set. An entry survives only until a
    /// proposal selects that genome as its parent, at which point the engine hands it over and removes it. Artifacts
    /// that were retained on an evaluation remain reachable through the archive entries in <see cref="Islands"/>. This
    /// text comes from evaluated candidates and is untrusted.
    /// </remarks>
    public IReadOnlyDictionary<string, IReadOnlyList<EvolutionArtifact>> PendingArtifacts { get; }

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
