using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Evolution;

/// <summary>The immutable result of an evolution run.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionRunResult<TGenome>
{
    /// <summary>Initializes a run result.</summary>
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
    public EvolutionArchiveEntry<TGenome>? Best => Islands.Select(archive => archive.Best)
        .OfType<EvolutionArchiveEntry<TGenome>>()
        .OrderBy(entry => entry.Evaluation.Quality,
            Islands.Count == 0 || Islands[0].Direction == EvolutionOptimizationDirection.Maximize
                ? Comparer<double?>.Create((x, y) => Nullable.Compare(y, x))
                : Comparer<double?>.Default)
        .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal)
        .FirstOrDefault();
}
