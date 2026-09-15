namespace AiDotNet.Evolution.Programs;

/// <summary>Identifies the shared ledger that enforces a metered program operator's proposal budget.</summary>
/// <remarks>The facade and portfolio validate reference identity, not merely matching unit labels, so separate
/// ledgers cannot silently give each child an independent copy of the advertised total budget.</remarks>
public interface IProgramResourceLedgerProvider
{
    /// <summary>Gets the live caller-owned ledger used for proposal admission and charging.</summary>
    EvolutionResourceLedger Ledger { get; }
}
