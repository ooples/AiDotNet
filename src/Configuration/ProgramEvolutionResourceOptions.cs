using AiDotNet.Evolution;

namespace AiDotNet.Configuration;

/// <summary>Shares a live ledger with program evaluation and optionally a metered proposal operator.</summary>
/// <remarks>
/// The evaluation maximum covers correctness checks and fitness together, in the declared common cost units.
/// This accounts for returned evaluator CostUnits, not arbitrary resource dimensions or uninstrumented model
/// work. The caller owns the ledger and must not charge the same evaluation elsewhere. Configuration clones
/// deliberately share this live ledger. Automatic engine checkpoint/resume is refused until ledger and engine
/// state can be committed together; a fresh ledger must not reset a resumed run's spending.
/// </remarks>
public sealed class ProgramEvolutionResourceOptions
{
    /// <summary>Creates validated evaluation accounting settings.</summary>
    public ProgramEvolutionResourceOptions(EvolutionResourceLedger ledger, decimal maximumEvaluationCostUnits, string costUnitVersionHash)
    {
        if (ledger is null) throw new ArgumentNullException(nameof(ledger));
        if (!ledger.Limits.Amounts.ContainsKey("cost_units"))
            throw new ArgumentException("The ledger must declare cost_units.", nameof(ledger));
        if (maximumEvaluationCostUnits <= 0 || maximumEvaluationCostUnits > EvolutionResources.MaximumAmount)
            throw new ArgumentOutOfRangeException(nameof(maximumEvaluationCostUnits));
        if (string.IsNullOrWhiteSpace(costUnitVersionHash) || costUnitVersionHash.Length > 256 || costUnitVersionHash.Any(char.IsControl))
            throw new ArgumentException("Declare bounded, printable common cost-unit semantics.", nameof(costUnitVersionHash));
        new System.Text.UTF8Encoding(false, true).GetByteCount(costUnitVersionHash);
        Ledger = ledger;
        MaximumEvaluationCostUnits = maximumEvaluationCostUnits;
        CostUnitVersionHash = costUnitVersionHash;
    }

    /// <summary>Gets the caller-owned ledger shared across instrumented stages.</summary>
    public EvolutionResourceLedger Ledger { get; }
    /// <summary>Gets the reserved maximum for one complete correctness-plus-fitness evaluation.</summary>
    public decimal MaximumEvaluationCostUnits { get; }
    /// <summary>Gets the declared conversion/measurement identity, which must match metered proposal credit.</summary>
    public string CostUnitVersionHash { get; }
}
