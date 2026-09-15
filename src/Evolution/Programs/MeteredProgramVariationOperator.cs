namespace AiDotNet.Evolution.Programs;

/// <summary>Adapts a cost-receipting program proposal backend for the facade and adaptive portfolios.</summary>
/// <remarks>The shared ledger enforces declared maxima before dispatch. Backend state and pending attribution
/// are checkpointed together; persist the matching ledger separately. This adapter does not enable automatic
/// facade resume or infer API pricing, token counts, compiler cost, or free work from missing usage.</remarks>
public sealed class MeteredProgramVariationOperator : IProgramVariationOperator,
    IOutcomeAwareVariationOperator<ProgramGenome>, IEvolutionProposalCostProvider, IProgramResourceLedgerProvider
{
    private readonly ICostedProgramProposalSource _source;
    private readonly ResourceMeteredVariationOperator<ProgramGenome> _metered;

    /// <summary>Creates an operator using explicit resource maxima and a common evaluator/proposal cost identity.</summary>
    public MeteredProgramVariationOperator(ICostedProgramProposalSource source, EvolutionResourceLedger ledger,
        EvolutionResources maximumProposalResources, string costUnitVersionHash)
    {
        _source = source ?? throw new ArgumentNullException(nameof(source));
        _metered = new(source, ledger, maximumProposalResources, costUnitVersionHash);
        Ledger = ledger;
    }

    /// <inheritdoc/>
    public string Id => _metered.Id;
    /// <inheritdoc/>
    public string VersionHash => _metered.VersionHash;
    /// <inheritdoc/>
    public string CostUnitVersionHash => _metered.CostUnitVersionHash;
    /// <inheritdoc/>
    public EvolutionResourceLedger Ledger { get; }
    /// <inheritdoc/>
    public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default) => _metered.ProposeAsync(context, cancellationToken);
    /// <inheritdoc/>
    public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) => _metered.Observe(evaluation, insertionResult);
    /// <inheritdoc/>
    public string CaptureState() => _metered.CaptureState();
    /// <inheritdoc/>
    public void RestoreState(string state) => _metered.RestoreState(state);
    /// <inheritdoc/>
    public EvolutionProposalCost GetProposalCost(long generation) => _metered.GetProposalCost(generation);
    /// <inheritdoc/>
    public ProgramEvolutionLlmUsage GetUsage() => _source.GetUsage();
}
