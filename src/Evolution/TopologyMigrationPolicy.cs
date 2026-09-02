using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Copies each island's best distinct elites along a configurable island topology.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The policy orders every source island's elites best-first, honoring the archive's
/// <see cref="EvolutionOptimizationDirection"/> and breaking quality ties by ordinal canonical genome ID, takes the
/// resolved number of migrants from the front of that order, and emits one transfer per destination named by the
/// configured <see cref="EvolutionMigrationTopology"/>. It only reads archive views; the engine applies the transfers,
/// marks each copy with its source island, and offers it to the destination archive under that archive's normal
/// insertion rules. Fewer than two islands yields no migrations, and although the contract supplies a
/// <see cref="StableRandom"/> the policy is fully deterministic and never consumes it, so a resumed run reproduces the
/// same transfers. Cost is <c>O(n * m log m)</c> for <c>n</c> islands holding at most <c>m</c> entries each.
/// </para>
/// <para>
/// The migrant count per source island is <c>migrantsPerIsland</c> when the rate is zero, and otherwise
/// <c>max(1, floor(eliteCount * rate))</c> capped at <c>migrantsPerIsland</c>, which is OpenEvolve's
/// <c>max(1, int(len(island_programs) * migration_rate))</c> rule (<c>database.py</c> <c>migrate_programs</c>) with an
/// explicit upper bound added so a large island cannot flood its neighbours. Enabling
/// <see cref="PreventsRepeatedMigration"/> additionally skips elites that arrived by an earlier migration, which is
/// OpenEvolve's hard-coded <c>metadata["migrant"]</c> guard made optional here because a MAP-Elites archive already
/// deduplicates by cell.
/// </para>
/// <para><b>For Beginners:</b> Islands are separate populations that explore the search space independently, and
/// migration is the step that copies a few of the best solutions from one island to another so good ideas can spread.
/// This policy decides which islands receive those copies. With the default <see cref="EvolutionMigrationTopology.Ring"/>
/// each island passes its best to the next island around a circle; with
/// <see cref="EvolutionMigrationTopology.FullyConnected"/> every island sends to every other, which spreads a winner in
/// one round but makes the islands look alike much sooner. The migration rate lets the number of travellers grow with
/// how full an island is instead of being a fixed count: a rate of <c>0.1</c> on an island holding 30 elites sends
/// three of them, still capped by the per-island maximum you configured. Use the defaults unless you have measured that
/// your islands converge too slowly.</para>
/// <para>
/// Island models and the effect of topology density on takeover time are analysed in Whitley, Rana, and Heckendorn,
/// "The Island Model Genetic Algorithm: On Separability, Population Size and Convergence" (Journal of Computing and
/// Information Technology, 1999).
/// </para>
/// </remarks>
public sealed class TopologyMigrationPolicy<TGenome> : IMigrationPolicy<TGenome>
{
    /// <summary>The fixed hub index used by <see cref="EvolutionMigrationTopology.Star"/>.</summary>
    private const int HubIsland = 0;

    private readonly string _versionHash;

    /// <summary>Initializes a topology-driven migration policy.</summary>
    /// <param name="topology">The island graph transfers follow; defaults to <see cref="EvolutionMigrationTopology.Ring"/>.</param>
    /// <param name="migrationRate">
    /// The fraction of a source island's elites to migrate, in the range <c>0</c> to <c>1</c>. Zero, the default,
    /// migrates exactly the caller's per-island maximum instead of a fraction.
    /// </param>
    /// <param name="preventRepeatedMigration">
    /// Whether elites that themselves arrived by migration are skipped as sources; <see langword="false"/>, the default,
    /// preserves the historical behaviour of migrating the best elites whatever their origin.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="topology"/> is undefined, or <paramref name="migrationRate"/> is not a finite value between zero
    /// and one inclusive.
    /// </exception>
    public TopologyMigrationPolicy(
        EvolutionMigrationTopology topology = EvolutionMigrationTopology.Ring,
        double migrationRate = 0,
        bool preventRepeatedMigration = false)
    {
        if (!Enum.IsDefined(typeof(EvolutionMigrationTopology), topology))
            throw new ArgumentOutOfRangeException(nameof(topology));
        if (double.IsNaN(migrationRate) || double.IsInfinity(migrationRate) || migrationRate < 0 || migrationRate > 1)
            throw new ArgumentOutOfRangeException(nameof(migrationRate),
                "The migration rate must be a finite fraction between zero and one.");
        Topology = topology;
        MigrationRate = migrationRate;
        PreventsRepeatedMigration = preventRepeatedMigration;
        _versionHash = string.Join("|", new[]
        {
            "topology-best-v1",
            ((int)topology).ToString(CultureInfo.InvariantCulture),
            migrationRate.ToString("R", CultureInfo.InvariantCulture),
            preventRepeatedMigration ? "no-remigration" : "remigration"
        });
    }

    /// <summary>Gets the island graph transfers follow.</summary>
    public EvolutionMigrationTopology Topology { get; }

    /// <summary>Gets the fraction of a source island's elites that migrate, or zero to use the per-island maximum.</summary>
    public double MigrationRate { get; }

    /// <summary>Gets whether elites that arrived by an earlier migration are skipped as migration sources.</summary>
    public bool PreventsRepeatedMigration { get; }

    /// <inheritdoc/>
    public string Id => "topology-best";

    /// <inheritdoc/>
    public string VersionHash => _versionHash;

    /// <inheritdoc/>
    public IReadOnlyList<EvolutionMigration<TGenome>> CreateMigrations(
        IReadOnlyList<IEvolutionArchiveView<TGenome>> islands,
        int migrantsPerIsland,
        StableRandom random)
    {
        Guard.NotNull(islands);
        Guard.Positive(migrantsPerIsland);
        Guard.NotNull(random);
        if (islands.Count < 2) return Array.Empty<EvolutionMigration<TGenome>>();

        var migrations = new List<EvolutionMigration<TGenome>>();
        for (int source = 0; source < islands.Count; source++)
        {
            IEvolutionArchiveView<TGenome> archive = islands[source];
            if (archive.Count == 0) continue;
            IReadOnlyList<int> destinations = DestinationsFor(Topology, source, islands.Count);
            if (destinations.Count == 0) continue;
            foreach (EvolutionArchiveEntry<TGenome> entry in SelectMigrants(archive, migrantsPerIsland))
                foreach (int destination in destinations)
                    migrations.Add(new EvolutionMigration<TGenome>(source, destination, entry));
        }
        return migrations;
    }

    /// <summary>Returns the distinct destination islands one source feeds under a topology.</summary>
    /// <param name="topology">The configured island graph.</param>
    /// <param name="source">The zero-based source island index.</param>
    /// <param name="islandCount">The total island count, which must be at least two.</param>
    /// <returns>The destinations in ascending order, never containing <paramref name="source"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="topology"/> is undefined, <paramref name="islandCount"/> is below two, or
    /// <paramref name="source"/> is outside the island range.
    /// </exception>
    /// <remarks>
    /// Exposed so a caller can predict, log, or test a topology's transfer set without running a migration round. The
    /// bidirectional ring collapses to a single destination when there are exactly two islands, because both
    /// neighbours are then the same island.
    /// </remarks>
    public static IReadOnlyList<int> DestinationsFor(EvolutionMigrationTopology topology, int source, int islandCount)
    {
        if (!Enum.IsDefined(typeof(EvolutionMigrationTopology), topology))
            throw new ArgumentOutOfRangeException(nameof(topology));
        if (islandCount < 2) throw new ArgumentOutOfRangeException(nameof(islandCount), "A topology needs at least two islands.");
        if (source < 0 || source >= islandCount) throw new ArgumentOutOfRangeException(nameof(source));

        var destinations = new List<int>();
        switch (topology)
        {
            case EvolutionMigrationTopology.BidirectionalRing:
                AddDestination(destinations, (source + 1) % islandCount, source);
                AddDestination(destinations, (source - 1 + islandCount) % islandCount, source);
                break;
            case EvolutionMigrationTopology.Star:
                if (source == HubIsland)
                {
                    for (int destination = 0; destination < islandCount; destination++)
                        AddDestination(destinations, destination, source);
                }
                else
                {
                    AddDestination(destinations, HubIsland, source);
                }
                break;
            case EvolutionMigrationTopology.FullyConnected:
                for (int destination = 0; destination < islandCount; destination++)
                    AddDestination(destinations, destination, source);
                break;
            default:
                AddDestination(destinations, (source + 1) % islandCount, source);
                break;
        }
        destinations.Sort();
        return destinations;
    }

    /// <summary>Resolves how many elites one source island contributes to a migration round.</summary>
    /// <param name="eliteCount">The number of elites currently held by the source island.</param>
    /// <param name="migrantsPerIsland">The positive per-island upper bound supplied by the engine.</param>
    /// <param name="migrationRate">The fraction of the island to migrate, or zero to use the bound directly.</param>
    /// <returns>The migrant count, always at least one and never above <paramref name="migrantsPerIsland"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="migrantsPerIsland"/> is not positive.</exception>
    /// <remarks>
    /// A rate of zero reproduces the fixed-count behaviour exactly, which is why raising the rate is the only way to
    /// change how many elites travel. A positive rate always sends at least one elite from a non-empty island, matching
    /// OpenEvolve's <c>max(1, ...)</c>, so a small island still participates.
    /// </remarks>
    public static int ResolveMigrantCount(int eliteCount, int migrantsPerIsland, double migrationRate)
    {
        Guard.Positive(migrantsPerIsland);
        if (eliteCount <= 0) return 0;
        if (migrationRate <= 0) return migrantsPerIsland;
        double scaled = eliteCount * migrationRate;
        if (scaled >= migrantsPerIsland) return migrantsPerIsland;
        int requested = (int)scaled;
        return requested < 1 ? 1 : requested;
    }

    private IEnumerable<EvolutionArchiveEntry<TGenome>> SelectMigrants(IEvolutionArchiveView<TGenome> archive,
        int migrantsPerIsland)
    {
        IEnumerable<EvolutionArchiveEntry<TGenome>> eligible = PreventsRepeatedMigration
            ? archive.Entries.Where(entry => !entry.Evaluation.Lineage.IsMigrant)
            : archive.Entries;
        EvolutionArchiveEntry<TGenome>[] candidates = eligible.ToArray();
        if (candidates.Length == 0) return Array.Empty<EvolutionArchiveEntry<TGenome>>();
        IOrderedEnumerable<EvolutionArchiveEntry<TGenome>> ordered =
            archive.Direction == EvolutionOptimizationDirection.Maximize
                ? candidates.OrderByDescending(entry => entry.Evaluation.Quality)
                    .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal)
                : candidates.OrderBy(entry => entry.Evaluation.Quality)
                    .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal);
        return ordered.Take(ResolveMigrantCount(candidates.Length, migrantsPerIsland, MigrationRate));
    }

    private static void AddDestination(List<int> destinations, int destination, int source)
    {
        if (destination == source || destinations.Contains(destination)) return;
        destinations.Add(destination);
    }
}
