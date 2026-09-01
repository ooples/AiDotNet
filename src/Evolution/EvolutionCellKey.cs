using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Immutable, value-based key for one multidimensional archive cell.</summary>
public sealed class EvolutionCellKey : IEquatable<EvolutionCellKey>, IComparable<EvolutionCellKey>
{
    private readonly int[] _bins;
    private readonly ReadOnlyCollection<int> _view;

    /// <summary>Initializes a key from one bin index per descriptor.</summary>
    /// <param name="bins">The nonempty bin sequence.</param>
    public EvolutionCellKey(IEnumerable<int> bins)
    {
        Guard.NotNull(bins);
        _bins = bins.ToArray();
        if (_bins.Length == 0) throw new ArgumentException("A cell key requires at least one bin.", nameof(bins));
        if (_bins.Any(value => value < 0)) throw new ArgumentOutOfRangeException(nameof(bins));
        _view = Array.AsReadOnly(_bins);
        StableKey = string.Join(",", _bins.Select(value => value.ToString(CultureInfo.InvariantCulture)));
    }

    /// <summary>Gets a read-only view of the bin indices.</summary>
    public IReadOnlyList<int> Bins => _view;

    /// <summary>Gets the culture-independent serialized key.</summary>
    public string StableKey { get; }

    /// <inheritdoc/>
    public bool Equals(EvolutionCellKey? other) => other is not null && StableKey == other.StableKey;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as EvolutionCellKey);

    /// <inheritdoc/>
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(StableKey);

    /// <inheritdoc/>
    public int CompareTo(EvolutionCellKey? other) => other is null ? 1 : StringComparer.Ordinal.Compare(StableKey, other.StableKey);

    /// <inheritdoc/>
    public override string ToString() => StableKey;
}
