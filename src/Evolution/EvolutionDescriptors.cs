using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Defines one named, bounded dimension of a quality-diversity archive.</summary>
public sealed class EvolutionDescriptorDefinition
{
    /// <summary>Initializes a fixed descriptor definition.</summary>
    /// <param name="name">The unique descriptor name.</param>
    /// <param name="minimum">The finite lower bound.</param>
    /// <param name="maximum">The finite upper bound, greater than <paramref name="minimum"/>.</param>
    /// <param name="binCount">The number of bins inside the configured bounds.</param>
    /// <param name="outOfRangePolicy">How values outside the bounds are handled.</param>
    public EvolutionDescriptorDefinition(
        string name,
        double minimum,
        double maximum,
        int binCount,
        EvolutionOutOfRangePolicy outOfRangePolicy = EvolutionOutOfRangePolicy.Reject)
    {
        Guard.NotNullOrWhiteSpace(name);
        if (!IsFinite(minimum)) throw new ArgumentOutOfRangeException(nameof(minimum), "The bound must be finite.");
        if (!IsFinite(maximum)) throw new ArgumentOutOfRangeException(nameof(maximum), "The bound must be finite.");
        if (maximum <= minimum) throw new ArgumentException("The maximum must be greater than the minimum.", nameof(maximum));
        if (!IsFinite(maximum - minimum)) throw new ArgumentOutOfRangeException(nameof(maximum), "The descriptor span must be finite.");
        Guard.Positive(binCount);
        if (outOfRangePolicy == EvolutionOutOfRangePolicy.OverflowBins && binCount > int.MaxValue - 2)
            throw new ArgumentOutOfRangeException(nameof(binCount));
        if (!Enum.IsDefined(typeof(EvolutionOutOfRangePolicy), outOfRangePolicy))
            throw new ArgumentOutOfRangeException(nameof(outOfRangePolicy));

        Name = name.Trim();
        Minimum = minimum;
        Maximum = maximum;
        BinCount = binCount;
        OutOfRangePolicy = outOfRangePolicy;
    }

    /// <summary>Gets the unique descriptor name.</summary>
    public string Name { get; }

    /// <summary>Gets the finite lower bound.</summary>
    public double Minimum { get; }

    /// <summary>Gets the finite upper bound.</summary>
    public double Maximum { get; }

    /// <summary>Gets the number of bins within the configured range.</summary>
    public int BinCount { get; }

    /// <summary>Gets the out-of-range policy.</summary>
    public EvolutionOutOfRangePolicy OutOfRangePolicy { get; }

    /// <summary>Gets the number of physical cells contributed by this dimension.</summary>
    public int EffectiveBinCount => OutOfRangePolicy == EvolutionOutOfRangePolicy.OverflowBins
        ? checked(BinCount + 2)
        : BinCount;

    /// <summary>Attempts to map a descriptor value to a stable bin index.</summary>
    /// <param name="value">The value to bin.</param>
    /// <param name="bin">The resulting zero-based bin when successful.</param>
    /// <returns><c>true</c> when the value is finite and accepted by the configured policy.</returns>
    public bool TryGetBin(double value, out int bin)
    {
        bin = -1;
        if (!IsFinite(value)) return false;

        if (value < Minimum)
        {
            if (OutOfRangePolicy == EvolutionOutOfRangePolicy.Reject) return false;
            bin = 0;
            return true;
        }

        if (value > Maximum)
        {
            if (OutOfRangePolicy == EvolutionOutOfRangePolicy.Reject) return false;
            bin = EffectiveBinCount - 1;
            return true;
        }

        double normalized = (value - Minimum) / (Maximum - Minimum);
        int interior = value == Maximum ? BinCount - 1 : (int)(normalized * BinCount);
        interior = Math.Max(0, Math.Min(BinCount - 1, interior));
        bin = OutOfRangePolicy == EvolutionOutOfRangePolicy.OverflowBins ? interior + 1 : interior;
        return true;
    }

    /// <summary>Returns a stable, culture-independent representation suitable for compatibility hashes.</summary>
    public string ToCanonicalString() => string.Join("|", new[]
    {
        Name.Length.ToString(CultureInfo.InvariantCulture) + ":" + Name,
        Minimum.ToString("R", CultureInfo.InvariantCulture),
        Maximum.ToString("R", CultureInfo.InvariantCulture),
        BinCount.ToString(CultureInfo.InvariantCulture),
        ((int)OutOfRangePolicy).ToString(CultureInfo.InvariantCulture)
    });

    internal static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}

/// <summary>
/// Collects order-independent observed bounds and freezes them into a fixed descriptor definition.
/// </summary>
public sealed class EvolutionDescriptorCalibrator
{
    private readonly string _name;
    private readonly int _binCount;
    private readonly EvolutionOutOfRangePolicy _policy;
    private double _minimum = double.PositiveInfinity;
    private double _maximum = double.NegativeInfinity;
    private long _observations;

    /// <summary>Initializes a descriptor calibrator.</summary>
    public EvolutionDescriptorCalibrator(
        string name,
        int binCount,
        EvolutionOutOfRangePolicy outOfRangePolicy = EvolutionOutOfRangePolicy.Clamp)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.Positive(binCount);
        if (!Enum.IsDefined(typeof(EvolutionOutOfRangePolicy), outOfRangePolicy))
            throw new ArgumentOutOfRangeException(nameof(outOfRangePolicy));
        _name = name.Trim();
        _binCount = binCount;
        _policy = outOfRangePolicy;
    }

    /// <summary>Gets the number of finite observations.</summary>
    public long ObservationCount => _observations;

    /// <summary>Adds one finite observation.</summary>
    /// <param name="value">The observed descriptor value.</param>
    public void Observe(double value)
    {
        if (!EvolutionDescriptorDefinition.IsFinite(value))
            throw new ArgumentOutOfRangeException(nameof(value), "Calibration values must be finite.");
        _minimum = Math.Min(_minimum, value);
        _maximum = Math.Max(_maximum, value);
        if (_observations == long.MaxValue) throw new InvalidOperationException("The calibration observation count overflowed.");
        _observations++;
    }

    /// <summary>Freezes the observed range into an immutable definition.</summary>
    /// <param name="relativePadding">Optional nonnegative fractional padding around the observed span.</param>
    public EvolutionDescriptorDefinition Freeze(double relativePadding = 0)
    {
        if (_observations == 0) throw new InvalidOperationException("At least one observation is required before freezing bounds.");
        if (!EvolutionDescriptorDefinition.IsFinite(relativePadding) || relativePadding < 0)
            throw new ArgumentOutOfRangeException(nameof(relativePadding));

        double minimum;
        double maximum;
        if (_minimum == _maximum)
        {
            if (_minimum == double.MaxValue)
            {
                minimum = NextDown(_minimum);
                maximum = _minimum;
            }
            else if (_minimum == -double.MaxValue)
            {
                minimum = _minimum;
                maximum = NextUp(_minimum);
            }
            else
            {
                minimum = NextDown(_minimum);
                maximum = NextUp(_maximum);
            }
        }
        else
        {
            double span = _maximum - _minimum;
            if (!EvolutionDescriptorDefinition.IsFinite(span))
                throw new InvalidOperationException("The observed descriptor span cannot be represented as a finite double.");
            double padding = span * relativePadding;
            minimum = _minimum - padding;
            maximum = _maximum + padding;
            if (!EvolutionDescriptorDefinition.IsFinite(minimum) || !EvolutionDescriptorDefinition.IsFinite(maximum))
                throw new InvalidOperationException("The padded descriptor bounds cannot be represented as finite doubles.");
        }
        return new EvolutionDescriptorDefinition(_name, minimum, maximum, _binCount, _policy);
    }

    private static double NextUp(double value)
    {
        if (value == 0) return double.Epsilon;
        long bits = BitConverter.DoubleToInt64Bits(value);
        return BitConverter.Int64BitsToDouble(value > 0 ? bits + 1 : bits - 1);
    }

    private static double NextDown(double value)
    {
        if (value == 0) return -double.Epsilon;
        long bits = BitConverter.DoubleToInt64Bits(value);
        return BitConverter.Int64BitsToDouble(value > 0 ? bits - 1 : bits + 1);
    }
}

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
