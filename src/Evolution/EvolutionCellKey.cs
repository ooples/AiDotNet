using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Immutable, value-based key for one multidimensional archive cell.</summary>
/// <remarks>
/// <para>
/// A quality-diversity archive discretizes each behavior descriptor into bins, and a cell is the combination of one
/// bin index per descriptor. This key stores that combination and exposes it two ways: <see cref="Bins"/> for
/// programmatic access and <see cref="StableKey"/>, a culture-independent comma-separated string such as
/// <c>"2,0,4"</c>, which serves as the dictionary key, the checkpoint representation, and the deterministic sort order
/// for archive entries. Equality, hashing, and comparison all delegate to <see cref="StableKey"/> with ordinal
/// semantics, so two keys built from equal bin sequences are interchangeable regardless of where they were created.
/// </para>
/// <para>
/// Bin indices must be nonnegative and the sequence must be nonempty; the constructor copies the input so later
/// changes to the source cannot affect the key. Construction is O(d) for d descriptors. Comparison is an ordinal
/// string comparison, so the order is lexicographic over the textual form rather than numeric per dimension (for
/// example <c>"10,0"</c> sorts before <c>"2,0"</c>); it is stable, which is what archives require.
/// </para>
/// <para><b>For Beginners:</b> Think of the archive as a spreadsheet where each row is one setting of the first
/// descriptor and each column is one setting of the second. A cell key is the coordinate of one square, like "row 2,
/// column 4", written in a form the computer can compare and store safely. You rarely build one yourself; the archive
/// computes it from a candidate's descriptor values and attaches it to each elite as
/// <see cref="EvolutionArchiveEntry{TGenome}.Cell"/>. You use it when you want to look up who currently occupies a
/// particular square with <see cref="AiDotNet.Interfaces.IEvolutionArchiveView{TGenome}.Get"/>, or when you print or
/// log which part of the map a result came from.</para>
/// </remarks>
public sealed class EvolutionCellKey : IEquatable<EvolutionCellKey>, IComparable<EvolutionCellKey>
{
    private readonly int[] _bins;
    private readonly ReadOnlyCollection<int> _view;

    /// <summary>Initializes a key from one bin index per descriptor.</summary>
    /// <param name="bins">The nonempty bin sequence.</param>
    /// <exception cref="ArgumentNullException"><paramref name="bins"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="bins"/> is empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A bin index is negative.</exception>
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
