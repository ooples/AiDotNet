using AiDotNet.Enums;
using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Mutable engine-owned quality-diversity archive contract.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionArchive<TGenome> : IEvolutionArchiveView<TGenome>
{
    /// <summary>Attempts to insert a completed candidate evaluation.</summary>
    EvolutionArchiveInsertionResult TryAdd(EvolutionCandidate<TGenome> candidate, EvolutionEvaluation evaluation);

    /// <summary>Samples uniformly from occupied cells using a caller-owned stable stream.</summary>
    EvolutionArchiveEntry<TGenome>? Sample(StableRandom random);
}
