using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Immutable ancestry and deterministic-stream metadata for a candidate.</summary>
public sealed class EvolutionLineage
{
    private readonly ReadOnlyCollection<string> _parentIds;
    private readonly ReadOnlyCollection<string> _inspirationIds;

    /// <summary>Initializes lineage metadata.</summary>
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
