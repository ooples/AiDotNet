namespace AiDotNet.Enums;

/// <summary>Selects the quantity an evolution run watches for a plateau before stopping early.</summary>
/// <remarks>
/// <para>
/// The engine evaluates the selected metric after every committed batch and normalises it so that larger is always
/// better, including under <see cref="EvolutionOptimizationDirection.Minimize"/>. A batch counts as an improvement only
/// when the metric gains at least <c>EvolutionEarlyStoppingOptions.MinimumImprovement</c>; otherwise the batch's
/// evaluations are added to the patience counter. OpenEvolve tracks only its <c>combined_score</c> metric and, when
/// patience is non-positive, compares floating-point values for exact equality (process_parallel.py:747-801); this
/// engine has no equality mode and can also watch archive occupancy, which is what actually stagnates in a
/// quality-diversity search whose best score is already saturated.
/// </para>
/// <para><b>For Beginners:</b> "Early stopping" means giving up when a search stops making progress instead of burning
/// the rest of the budget. The question is: progress at what? <see cref="BestQuality"/> watches the single best score,
/// which is the familiar choice. <see cref="Coverage"/> watches how much of the behaviour map has been filled, which is
/// the right choice when you care about getting a varied set of solutions rather than one champion.
/// <see cref="QdScore"/> watches both at once by adding up the scores of every cell, so it improves either when a cell
/// gets better or when a new cell is filled.</para>
/// </remarks>
public enum EvolutionEarlyStoppingMetric
{
    /// <summary>The best elite quality across every island, negated under minimization.</summary>
    BestQuality = 0,
    /// <summary>The fraction of grid cells occupied across every island.</summary>
    Coverage = 1,
    /// <summary>The sum of every elite's quality across every island, negated under minimization.</summary>
    QdScore = 2
}
