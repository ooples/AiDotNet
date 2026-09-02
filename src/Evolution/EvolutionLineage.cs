using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Immutable ancestry and deterministic-stream metadata for a candidate.</summary>
/// <remarks>
/// <para>
/// Lineage records how a candidate came to exist: the canonical IDs of its parent and inspiration elites, which
/// variation operator and optional refiner produced it, the logical generation and island it was proposed for, and the
/// <see cref="SeedStream"/> identifier from which its deterministic random streams were derived. Seed genomes carry an
/// empty parent list and the operator ID <c>"seed"</c>. The engine attaches lineage to every evaluation and archive
/// entry, serializes it into checkpoints, and includes it in the run state hash, so IDs are trimmed and validated on
/// construction to keep those hashes stable.
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
    /// <exception cref="ArgumentException">An identity collection contains an empty value, or <paramref name="variationOperatorId"/> is empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="generation"/> or <paramref name="island"/> is negative.</exception>
    public EvolutionLineage(
        IEnumerable<string>? parentIds,
        IEnumerable<string>? inspirationIds,
        string variationOperatorId,
        string? refinerId,
        long generation,
        int island,
        ulong seedStream)
    {
        Guard.NotNullOrWhiteSpace(variationOperatorId);
        if (generation < 0) throw new ArgumentOutOfRangeException(nameof(generation));
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        _parentIds = Array.AsReadOnly(CopyIds(parentIds, nameof(parentIds)));
        _inspirationIds = Array.AsReadOnly(CopyIds(inspirationIds, nameof(inspirationIds)));
        VariationOperatorId = variationOperatorId.Trim();
        RefinerId = refinerId is null || string.IsNullOrWhiteSpace(refinerId) ? null : refinerId.Trim();
        Generation = generation;
        Island = island;
        SeedStream = seedStream;
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

    private static string[] CopyIds(IEnumerable<string>? values, string parameterName)
    {
        if (values is null) return Array.Empty<string>();
        string[] result = values.ToArray();
        if (result.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Identity collections cannot contain empty values.", parameterName);
        return result.Select(value => value.Trim()).ToArray();
    }
}
