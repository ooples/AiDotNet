using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>Configures how descriptor bounds are derived from what a seed population actually measured.</summary>
/// <remarks>
/// <para>
/// A quality-diversity archive is a grid, and a grid needs bounds. Choosing them by hand is the one piece of setup
/// that genuinely cannot be guessed from the code: it depends on what the descriptors mean and what values the
/// problem produces. Get them wrong in one direction and every candidate lands in one cell, so the search collapses
/// to a plain optimizer; wrong in the other and the grid is mostly empty, so nothing ever competes.
/// </para>
/// <para>
/// Calibration removes that step by reading the answer off the seed population. The seeds are a fixed, ordered set
/// the caller already supplies, so the bounds derived from them are a pure function of the run's own inputs — the
/// same seeds give the same grid on any machine, and the grid is recorded in the archive's definition hash like any
/// hand-written one. Later candidates that fall outside the seeded span widen the grid in whole bins under
/// <see cref="OutOfRangePolicy"/> rather than being discarded.
/// </para>
/// <para><b>For Beginners:</b> Name the behaviours you care about and let this pick the grid. The seeds you already
/// hand the search are measured once, and the range they cover — plus a margin — becomes the grid. If later
/// candidates go outside it, the grid grows to fit them.</para>
/// </remarks>
public sealed class EvolutionDescriptorCalibrationOptions
{
    /// <summary>Gets or sets how many bins each calibrated axis is divided into.</summary>
    /// <remarks>
    /// This is the resolution of the search's diversity, and it multiplies: with three axes, 16 bins each is 4,096
    /// cells. Prefer fewer bins on more axes over more bins on fewer.
    /// </remarks>
    public int BinCount { get; set; } = 16;

    /// <summary>Gets or sets how much of the observed span is added at each end, as a fraction of that span.</summary>
    /// <remarks>
    /// Seeds rarely bracket the range the search will reach. A margin keeps the best and worst seeds off the very
    /// edge of the grid, so the first children that improve on them have somewhere to land without the grid having
    /// to grow immediately. Zero means the grid covers exactly the seeded span.
    /// </remarks>
    public double Padding { get; set; } = 0.25;

    /// <summary>Gets or sets the span used when every seed reported the same value for an axis.</summary>
    /// <remarks>
    /// A zero-width axis cannot be binned at all, and it is a real case: a descriptor that starts at zero for every
    /// seed and only moves once the search finds something. The axis becomes this wide, centred on the shared value,
    /// and grows from there.
    /// </remarks>
    public double DegenerateSpan { get; set; } = 1.0;

    /// <summary>Gets or sets what happens to a value outside the calibrated bounds.</summary>
    /// <remarks>
    /// Defaults to <see cref="EvolutionOutOfRangePolicy.Grow"/>, which is the point of calibrating: the seeds fix a
    /// sensible bin width, and anything the search later reaches beyond them extends the grid in whole bins of that
    /// same width rather than being clamped into the end cell or rejected outright.
    /// </remarks>
    public EvolutionOutOfRangePolicy OutOfRangePolicy { get; set; } = EvolutionOutOfRangePolicy.Grow;

    /// <summary>Creates an independent copy so a running calibration is unaffected by later mutation.</summary>
    /// <returns>A new options instance carrying the same values.</returns>
    public EvolutionDescriptorCalibrationOptions Clone() => new()
    {
        BinCount = BinCount,
        Padding = Padding,
        DegenerateSpan = DegenerateSpan,
        OutOfRangePolicy = OutOfRangePolicy
    };

    /// <summary>Validates the bin count, the margin, and the degenerate span.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A value is outside its permitted range.</exception>
    public void Validate()
    {
        if (BinCount < 1 || BinCount > 100_000)
            throw new ArgumentOutOfRangeException(nameof(BinCount), BinCount, "Value must be between 1 and 100000.");
        if (double.IsNaN(Padding) || double.IsInfinity(Padding) || Padding < 0 || Padding > 100)
            throw new ArgumentOutOfRangeException(nameof(Padding), Padding, "Value must be between 0 and 100.");
        if (double.IsNaN(DegenerateSpan) || double.IsInfinity(DegenerateSpan) || DegenerateSpan <= 0)
            throw new ArgumentOutOfRangeException(nameof(DegenerateSpan), DegenerateSpan, "Value must be finite and positive.");
        if (!Enum.IsDefined(typeof(EvolutionOutOfRangePolicy), OutOfRangePolicy))
            throw new ArgumentOutOfRangeException(nameof(OutOfRangePolicy), OutOfRangePolicy, "Value must be a defined policy.");
    }
}
