using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Defines one named, bounded dimension of a quality-diversity archive.</summary>
/// <remarks>
/// <para>
/// A MAP-Elites archive is a grid, and each <see cref="EvolutionDescriptorDefinition"/> is one axis of that grid.
/// The definition fixes the axis <see cref="Name"/>, its finite <see cref="Minimum"/> and <see cref="Maximum"/>,
/// the number of equal-width <see cref="BinCount"/> cells between them, and what happens to a value outside the
/// bounds: <see cref="EvolutionOutOfRangePolicy.Reject"/> makes <see cref="TryGetBin"/> return <c>false</c> so the
/// candidate is not archived, <see cref="EvolutionOutOfRangePolicy.Clamp"/> folds the value into the first or last
/// interior bin, and <see cref="EvolutionOutOfRangePolicy.OverflowBins"/> reserves one extra cell below and one
/// above the range, which is why <see cref="EffectiveBinCount"/> can exceed <see cref="BinCount"/>.
/// <see cref="ToCanonicalString"/> is folded into the archive definition hash and therefore into checkpoint
/// compatibility, so changing any bound or bin count invalidates older checkpoints by design.
/// </para>
/// <para><b>For Beginners:</b> Quality-diversity search does not keep only the single best solution; it keeps the
/// best solution for every "kind" of behaviour, and descriptors are how you tell the engine which kinds matter.
/// Suppose you are evolving small neural networks and care about model size and latency: you would define two
/// descriptors, <c>"parameters"</c> from 1,000 to 1,000,000 in 10 bins and <c>"latencyMs"</c> from 0 to 50 in 5
/// bins, giving a 10 x 5 grid in which each cell remembers the most accurate network of that size and speed. A
/// descriptor is like one axis of a chart with a fixed number of tick marks. Choose bounds that cover the values you
/// expect to see; use <see cref="EvolutionOutOfRangePolicy.Clamp"/> or
/// <see cref="EvolutionOutOfRangePolicy.OverflowBins"/> if surprising values should still be archived, or keep the
/// default <see cref="EvolutionOutOfRangePolicy.Reject"/> if a value outside the range indicates a broken
/// candidate.</para>
/// <para>
/// Background: Mouret &amp; Clune (2015), "Illuminating search spaces by mapping elites", arXiv:1504.04909. Binning
/// is O(1): the interior index is <c>floor((value - Minimum) / (Maximum - Minimum) * BinCount)</c>, with the exact
/// <see cref="Maximum"/> assigned to the last interior bin, and every computation uses the invariant culture so
/// results are identical on every machine.
/// </para>
/// </remarks>
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
