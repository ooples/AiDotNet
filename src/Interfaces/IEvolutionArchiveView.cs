using AiDotNet.Enums;
using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Read-only view of a deterministic quality-diversity archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A quality-diversity archive such as <see cref="MapElitesArchive{TGenome}"/> is a grid of cells, one per
/// combination of descriptor bins, that keeps the single best evaluated candidate found so far in each cell. This
/// interface exposes everything a reader needs, namely the descriptor definitions, the elites in stable cell-key
/// order, the globally best elite, and a per-cell lookup, without exposing any operation that could mutate the
/// archive. <see cref="Version"/> increases on every accepted insertion or replacement, so callers can detect change
/// cheaply, and <see cref="DefinitionHash"/> identifies the descriptor, direction, and capacity policies so that a
/// checkpoint is only restored into a compatible archive.
/// </para>
/// <para>
/// Ordering is deterministic everywhere: <see cref="Entries"/> is sorted by
/// <see cref="EvolutionCellKey.StableKey"/> and <see cref="Best"/> breaks quality ties by canonical genome id, so two
/// runs with the same seed produce identical views. Live archives are owned by a single writer inside the engine; to
/// hold a view across engine steps, copy it with <see cref="EvolutionArchiveSnapshot{TGenome}"/>, which is also what
/// <see cref="EvolutionRunResult{TGenome}.Islands"/> returns once a run finishes.
/// </para>
/// <para><b>For Beginners:</b> Imagine a trophy case with one shelf per category, where each shelf holds only the
/// best trophy earned in that category. This interface lets you look into the case (count the shelves in use, read
/// each trophy, find the best one overall) but never lets you move anything. Use it when you want to inspect or
/// report results, for example to print a table of the best model per model family after a MAP-Elites AutoML run,
/// without risking a change to the search state. This diverse map of winners, rather than a single winner, is what
/// makes quality-diversity search different from an ordinary optimizer.</para>
/// </remarks>
public interface IEvolutionArchiveView<TGenome>
{
    /// <summary>Gets immutable descriptor definitions.</summary>
    IReadOnlyList<EvolutionDescriptorDefinition> Descriptors { get; }

    /// <summary>Gets a stable hash of every archive policy that affects insertion or restoration.</summary>
    string DefinitionHash { get; }

    /// <summary>Gets the configured optimization direction.</summary>
    EvolutionOptimizationDirection Direction { get; }

    /// <summary>Gets the current number of occupied cells.</summary>
    int Count { get; }

    /// <summary>Gets a monotonically increasing change version.</summary>
    long Version { get; }

    /// <summary>Gets elites in stable cell-key order.</summary>
    IReadOnlyList<EvolutionArchiveEntry<TGenome>> Entries { get; }

    /// <summary>Gets the globally best elite with deterministic tie-breaking.</summary>
    EvolutionArchiveEntry<TGenome>? Best { get; }

    /// <summary>Returns the elite in a specific cell, or <c>null</c>.</summary>
    /// <param name="cell">The cell to look up.</param>
    /// <returns>The occupying elite, or <c>null</c> when the cell is empty.</returns>
    EvolutionArchiveEntry<TGenome>? Get(EvolutionCellKey cell);
}
