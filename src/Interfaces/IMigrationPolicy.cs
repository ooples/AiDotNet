using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Creates deterministic elite transfers between independent island archives.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The evolution engine calls <see cref="CreateMigrations"/> once every configured number of committed
/// batches (<c>EvolutionEngineOptions.MigrationInterval</c>) when more than one island is configured. A
/// policy only proposes transfers and must not mutate any archive. The engine validates every proposal
/// (non-null transfers, distinct in-range island indices, at most <c>migrantsPerIsland</c> transfers per
/// source island, and an entry that is genuinely present in its source island), orders the transfers
/// deterministically by source, destination, and genome identifier, and then offers each migrant to its
/// destination archive through the normal insertion rules, so a migrant that does not beat the destination's
/// incumbent is simply not copied. Any randomness must come from the supplied <see cref="StableRandom"/>
/// stream so that resumed runs reproduce the same transfers. <see cref="Id"/> and <see cref="VersionHash"/>
/// are folded into the checkpoint compatibility hash, so swapping or revising a policy prevents an older
/// checkpoint from being resumed under different migration semantics.
/// </para>
/// <para><b>For Beginners:</b> An island model runs several separate populations (islands) side by side so
/// that each one can explore a different part of the search space without being taken over by a single
/// early winner. Every so often a few of the best solutions are copied from one island to another; that copy
/// is called a migration, and this interface decides which solutions move and where they go, much like
/// independent research teams that periodically share their best results with a neighbouring team. The
/// built-in <see cref="RingMigrationPolicy{TGenome}"/> copies each island's top elites to the next island in a
/// ring and is a good default. Implement this interface yourself only when you need a different topology,
/// such as broadcasting to every island or migrating random rather than best elites. Because the engine hands
/// you a seeded random stream, any randomness you use stays reproducible and checkpoint-safe.</para>
/// <para>
/// Island models are analysed in Whitley, Rana, and Heckendorn, "The Island Model Genetic Algorithm: On
/// Separability, Population Size and Convergence" (Journal of Computing and Information Technology, 1999).
/// </para>
/// </remarks>
public interface IMigrationPolicy<TGenome>
{
    /// <summary>Gets a stable policy identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    /// <remarks>
    /// Change this value whenever the transfer semantics change so that a checkpoint written under the old
    /// behaviour is not resumed under the new one.
    /// </remarks>
    string VersionHash { get; }

    /// <summary>Creates transfers without modifying any archive.</summary>
    /// <param name="islands">Read-only views of every island archive, indexed by island number.</param>
    /// <param name="migrantsPerIsland">The maximum number of transfers that may originate from any one island.</param>
    /// <param name="random">A caller-owned deterministic stream for any randomized selection.</param>
    /// <returns>The proposed transfers, or an empty list when nothing should migrate.</returns>
    IReadOnlyList<EvolutionMigration<TGenome>> CreateMigrations(
        IReadOnlyList<IEvolutionArchiveView<TGenome>> islands,
        int migrantsPerIsland,
        StableRandom random);
}
