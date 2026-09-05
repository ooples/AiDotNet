using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Immutable ancestry and deterministic-stream metadata for a candidate.</summary>
/// <remarks>
/// <para>
/// Lineage records how a candidate came to exist: the canonical IDs of its parent and inspiration elites, which
/// variation operator and optional refiner produced it, the logical generation and island it was proposed for, and the
/// <see cref="SeedStream"/> identifier from which its deterministic random streams were derived. Seed genomes carry an
/// empty parent list and the operator ID <c>"seed"</c>. An elite copied between islands by a migration round carries a
/// <see cref="MigrationSourceIsland"/>, which is what distinguishes it from a candidate discovered locally on the same
/// island. The engine attaches lineage to every evaluation and archive entry, serializes it into checkpoints, and
/// includes it in the run state hash, so IDs are trimmed and validated on construction to keep those hashes stable.
/// </para>
/// <para><b>For Beginners:</b> This is a candidate's family tree written on a card: who its parents were, which
/// operator created it, in which generation and on which island it appeared, and which random stream was used. You do
/// not normally create one yourself; the engine attaches it to every candidate so that you can later trace how a strong
/// solution was found, for example by following <see cref="ParentIds"/> back through the archive to the original seed.
/// Seeds are easy to spot because their <see cref="ParentIds"/> list is empty and their operator ID is <c>"seed"</c>.
/// It is also part of what makes a run reproducible: given the lineage and the run seed, the exact random draws that
/// produced the candidate can be regenerated.</para>
/// </remarks>
public sealed class EvolutionLineage
{
    private readonly ReadOnlyCollection<string> _parentIds;
    private readonly ReadOnlyCollection<string> _inspirationIds;

    /// <summary>Initializes lineage metadata.</summary>
    /// <param name="parentIds">Canonical IDs of direct parents, or <see langword="null"/> for a seed.</param>
    /// <param name="inspirationIds">Canonical IDs of inspiration elites, or <see langword="null"/> when none were used.</param>
    /// <param name="variationOperatorId">The non-empty identifier of the operator that produced the candidate.</param>
    /// <param name="refinerId">The optional refiner identifier; empty or whitespace values are stored as <see langword="null"/>.</param>
    /// <param name="generation">The non-negative logical generation.</param>
    /// <param name="island">The non-negative zero-based island index.</param>
    /// <param name="seedStream">The deterministic random stream identifier.</param>
    /// <param name="migrationSourceIsland">
    /// The island this elite was copied from by a migration round, or <see langword="null"/> (the default) when the
    /// candidate was discovered locally on <paramref name="island"/>.
    /// </param>
    /// <exception cref="ArgumentException">
    /// An identity collection contains an empty value, <paramref name="variationOperatorId"/> is empty, or
    /// <paramref name="migrationSourceIsland"/> equals <paramref name="island"/>.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="generation"/>, <paramref name="island"/>, or <paramref name="migrationSourceIsland"/> is negative.
    /// </exception>
    public EvolutionLineage(
        IEnumerable<string>? parentIds,
        IEnumerable<string>? inspirationIds,
        string variationOperatorId,
        string? refinerId,
        long generation,
        int island,
        ulong seedStream,
        int? migrationSourceIsland = null)
    {
        Guard.NotNullOrWhiteSpace(variationOperatorId);
        if (generation < 0) throw new ArgumentOutOfRangeException(nameof(generation));
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        if (migrationSourceIsland.HasValue && migrationSourceIsland.Value < 0)
            throw new ArgumentOutOfRangeException(nameof(migrationSourceIsland));
        if (migrationSourceIsland.HasValue && migrationSourceIsland.Value == island)
            throw new ArgumentException("A migrated elite cannot name its destination island as its source.",
                nameof(migrationSourceIsland));
        _parentIds = Array.AsReadOnly(CopyIds(parentIds, nameof(parentIds)));
        _inspirationIds = Array.AsReadOnly(CopyIds(inspirationIds, nameof(inspirationIds)));
        VariationOperatorId = variationOperatorId.Trim();
        RefinerId = refinerId is null || string.IsNullOrWhiteSpace(refinerId) ? null : refinerId.Trim();
        Generation = generation;
        Island = island;
        SeedStream = seedStream;
        MigrationSourceIsland = migrationSourceIsland;
    }

    /// <summary>Gets canonical IDs of direct parents.</summary>
    public IReadOnlyList<string> ParentIds => _parentIds;

    /// <summary>Gets canonical IDs supplied as variation inspirations.</summary>
    public IReadOnlyList<string> InspirationIds => _inspirationIds;

    /// <summary>Gets the variation operator identifier.</summary>
    public string VariationOperatorId { get; }

    /// <summary>Gets the optional refiner identifier.</summary>
    public string? RefinerId { get; }

    /// <summary>Gets the logical generation number.</summary>
    public long Generation { get; }

    /// <summary>Gets the zero-based island index.</summary>
    public int Island { get; }

    /// <summary>Gets the deterministic random stream identifier.</summary>
    public ulong SeedStream { get; }

    /// <summary>Gets the island a migration round copied this elite from, or <c>null</c> for a local discovery.</summary>
    /// <remarks>
    /// The engine sets this only on the copy it offers to a destination archive during migration; the original entry in
    /// the source island keeps its own lineage untouched. Because the value is part of the persisted lineage it
    /// survives a checkpoint round trip and is folded into the run state hash, unlike OpenEvolve's
    /// <c>metadata["migrant"]</c> flag, which is only attached to the in-memory copy.
    /// </remarks>
    public int? MigrationSourceIsland { get; }

    /// <summary>Gets whether this record describes an elite copied into its island by a migration round.</summary>
    public bool IsMigrant => MigrationSourceIsland.HasValue;

    private static string[] CopyIds(IEnumerable<string>? values, string parameterName)
    {
        if (values is null) return Array.Empty<string>();
        string[] result = values.ToArray();
        if (result.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Identity collections cannot contain empty values.", parameterName);
        return result.Select(value => value.Trim()).ToArray();
    }
}
