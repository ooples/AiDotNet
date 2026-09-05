namespace AiDotNet.Enums;

/// <summary>Controls how descriptor values outside configured bounds are binned.</summary>
/// <remarks>
/// <para>
/// Each <see cref="AiDotNet.Evolution.EvolutionDescriptorDefinition"/> divides its finite
/// [<c>Minimum</c>, <c>Maximum</c>] range into <c>BinCount</c> equal-width bins. This policy decides what happens
/// when an evaluator reports a descriptor value below the minimum or above the maximum. <see cref="Reject"/> makes
/// the candidate un-binnable, so the archive answers with <see cref="EvolutionArchiveInsertionResult.Rejected"/>;
/// <see cref="Clamp"/> folds out-of-range values into the first or last interior bin without growing the grid;
/// <see cref="OverflowBins"/> adds one extra bin on each side, so the dimension contributes <c>BinCount + 2</c>
/// physical cells and out-of-range candidates are archived separately from in-range ones. Non-finite values (NaN
/// or infinity) are rejected under every policy. The choice is part of the archive definition hash, so changing
/// it invalidates existing checkpoints.
/// </para>
/// <para><b>For Beginners:</b> A quality-diversity archive is like a grid of shelves where each descriptor (for
/// example "model size") picks the row, and only the best candidate found for that row is kept on the shelf
/// (MAP-Elites: Mouret and Clune, 2015). You decide the size range up front, but real candidates sometimes land
/// outside it. <see cref="Reject"/> simply discards those candidates, which is right when out-of-range means
/// "invalid". <see cref="Clamp"/> puts them on the nearest edge shelf, which keeps them but lets them compete with
/// genuinely in-range candidates for that shelf. <see cref="OverflowBins"/> gives them their own "too small" and
/// "too large" shelves, so you keep the information without distorting the interior grid; use it when the range
/// you configured is a rough guess.</para>
/// </remarks>
public enum EvolutionOutOfRangePolicy
{
    /// <summary>Reject the candidate when a descriptor is outside its configured range.</summary>
    Reject = 0,
    /// <summary>Clamp below-range and above-range values into the first and last bins.</summary>
    Clamp = 1,
    /// <summary>Reserve explicit bins below and above the configured range.</summary>
    OverflowBins = 2,
    /// <summary>Extend the range in whole bins so the value becomes interior, re-keying existing entries.</summary>
    /// <remarks>
    /// <para>
    /// The bin width never changes, so a cell means the same thing before and after growth and every existing
    /// elite is re-keyed by arithmetic rather than re-evaluated. Growth is driven only by committed evaluations in
    /// the engine's deterministic commit order, so a replayed or resumed run grows identically; the grown bounds
    /// travel in the checkpoint, while the compatibility hash keeps the ORIGINAL bounds so a resume is not refused
    /// by the archive having done its job.
    /// </para>
    /// <para><b>For Beginners:</b> Choose this when you genuinely cannot guess the range up front. The grid adds
    /// whole rows of identically-sized pigeonholes as it discovers wider values, instead of discarding those
    /// candidates or squashing them onto an edge shelf.</para>
    /// </remarks>
    Grow = 3
}
