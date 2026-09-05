namespace AiDotNet.Enums;

/// <summary>Selects the unit the evolution engine counts before running an island migration round.</summary>
/// <remarks>
/// <para>
/// <see cref="CommittedBatches"/> counts committed logical batches, so a migration happens every
/// <c>MigrationInterval</c> batches regardless of how many proposals each batch contained. <see cref="IslandGenerations"/>
/// instead compares the highest per-island generation counter against the generation at which the previous migration
/// ran, which is the rule OpenEvolve's program database uses. Both counters are rolled back with an interrupted batch
/// transaction, so cancelling a run can never leave a migration half-scheduled, and the selected trigger is folded
/// into the engine configuration hash.
/// </para>
/// <para><b>For Beginners:</b> Islands periodically send copies of their best solutions to their neighbours so good
/// ideas can spread. This setting decides how the engine measures "periodically". Counting batches is the simplest
/// and most predictable option: every N rounds of work, migrate. Counting generations instead follows how far the
/// busiest island has actually advanced, which matters when some batches contain far more new candidates than
/// others. If you are unsure, leave it on <see cref="CommittedBatches"/>.</para>
/// </remarks>
public enum EvolutionMigrationTrigger
{
    /// <summary>Migrate once every <c>MigrationInterval</c> committed logical batches.</summary>
    CommittedBatches = 0,

    /// <summary>Migrate once the highest per-island generation has advanced by <c>MigrationInterval</c>.</summary>
    IslandGenerations = 1
}
