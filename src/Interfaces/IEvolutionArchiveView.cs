using AiDotNet.Enums;
using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Read-only view of a deterministic quality-diversity archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
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
    EvolutionArchiveEntry<TGenome>? Get(EvolutionCellKey cell);

}
