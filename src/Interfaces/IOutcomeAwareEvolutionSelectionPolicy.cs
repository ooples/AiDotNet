using AiDotNet.Enums;
using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Optional state and outcome feedback implemented by adaptive selection policies.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IOutcomeAwareEvolutionSelectionPolicy<TGenome> : ISelectionPolicy<TGenome>
{
    /// <summary>Updates policy state after one evaluation is committed.</summary>
    void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult);

    /// <summary>Captures deterministic policy state for a checkpoint.</summary>
    string CaptureState();

    /// <summary>Restores deterministic policy state from a checkpoint.</summary>
    void RestoreState(string state);
}
