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

        // Grow reports failure here exactly as Reject does, because widening the range is the archive's decision:
        // it owns the existing entries that a wider grid re-keys, and this method must stay a pure function of the
        // definition it is called on.
        if (value < Minimum)
        {
            if (OutOfRangePolicy is EvolutionOutOfRangePolicy.Reject or EvolutionOutOfRangePolicy.Grow) return false;
            bin = 0;
            return true;
        }

        if (value > Maximum)
        {
            if (OutOfRangePolicy is EvolutionOutOfRangePolicy.Reject or EvolutionOutOfRangePolicy.Grow) return false;
            bin = EffectiveBinCount - 1;
            return true;
        }

        double normalized = (value - Minimum) / (Maximum - Minimum);
        int interior = value == Maximum ? BinCount - 1 : (int)(normalized * BinCount);
        interior = Math.Max(0, Math.Min(BinCount - 1, interior));
        bin = OutOfRangePolicy == EvolutionOutOfRangePolicy.OverflowBins ? interior + 1 : interior;
        return true;
    }

    /// <summary>Gets the width of one interior bin.</summary>
    public double BinWidth => (Maximum - Minimum) / BinCount;

    /// <summary>Bins a value as if the range contained it, whatever the out-of-range policy says.</summary>
    /// <param name="value">The value to bin.</param>
    /// <param name="bin">The resulting zero-based bin when the value is inside the range.</param>
    /// <returns><c>true</c> when the value is finite and inside the range.</returns>
    /// <remarks>
    /// Used by a growing archive to check that a widened definition really accepts the value it was widened for,
    /// before adopting it. <see cref="TryGetBin"/> cannot answer that question, because under
    /// <see cref="EvolutionOutOfRangePolicy.Grow"/> it deliberately reports every out-of-range value as unbinnable.
    /// </remarks>
    internal bool TryGetBinIgnoringPolicy(double value, out int bin)
    {
        bin = -1;
        if (!IsFinite(value) || value < Minimum || value > Maximum) return false;

        double normalized = (value - Minimum) / (Maximum - Minimum);
        int interior = value == Maximum ? BinCount - 1 : (int)(normalized * BinCount);
        bin = Math.Max(0, Math.Min(BinCount - 1, interior));
        return true;
    }

    /// <summary>Returns a definition widened to contain <paramref name="value"/>, keeping the bin width fixed.</summary>
    /// <param name="value">The finite value the widened definition must accept.</param>
    /// <returns>The widened definition, or this instance when the value already fits or cannot be reached.</returns>
    /// <remarks>
    /// <para>
    /// Holding the bin width constant is what makes growth safe: a cell covers the same span of values before and
    /// after, so an archive can move its existing elites onto the wider grid by rebinning them rather than
    /// re-evaluating them. Only whole bins are added, and only as many as the value needs, which makes growth
    /// order-independent: widening to reach one value and then a more extreme one lands on the same bounds as
    /// widening straight to the more extreme one.
    /// </para>
    /// <para><b>For Beginners:</b> The archive is a grid of pigeonholes. This adds whole rows of pigeonholes at one
    /// end, all the same size as the existing ones, so nothing already filed has to be re-measured.</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="value"/> is not finite.</exception>
    public EvolutionDescriptorDefinition Widen(double value)
    {
        if (!IsFinite(value)) throw new ArgumentOutOfRangeException(nameof(value), "The value must be finite.");
        if (value >= Minimum && value <= Maximum) return this;

        double width = BinWidth;
        if (!IsFinite(width) || width <= 0) return this;

        double minimum = Minimum;
        double maximum = Maximum;
        long bins = BinCount;

        if (value < minimum)
        {
            long needed = Math.Max(1, (long)Math.Ceiling((minimum - value) / width));
            minimum -= needed * width;
            bins += needed;
        }
        else
        {
            long needed = Math.Max(1, (long)Math.Ceiling((value - maximum) / width));
            maximum += needed * width;
            bins += needed;
        }

        if (bins > int.MaxValue / 2 || !IsFinite(minimum) || !IsFinite(maximum) || !IsFinite(maximum - minimum))
            return this;

        return new EvolutionDescriptorDefinition(Name, minimum, maximum, (int)bins, OutOfRangePolicy);
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
