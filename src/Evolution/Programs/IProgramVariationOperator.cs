namespace AiDotNet.Evolution.Programs;

/// <summary>A consumer-supplied program proposal loop with explicit language-model usage reporting.</summary>
/// <remarks>
/// Configure through ProgramEvolutionOptions.CustomVariation. The operator owns its prompting, compiler,
/// repair, edit-boundary enforcement, audit trail and provider lifecycle. It must never consume hidden-test
/// feedback. Implement ICheckpointableVariationOperator when requesting engine checkpointing, and
/// IEvolutionProposalCostProvider when supplying metered proposal credit. A usage count of zero means no
/// reported work, not evidence that a provider call was free. Use a fresh operator for an independent run.
/// </remarks>
public interface IProgramVariationOperator : IVariationOperator<ProgramGenome>
{
    /// <summary>Returns observed model work, including failed attempts and repairs, without resetting counters.</summary>
    ProgramEvolutionLlmUsage GetUsage();
}
