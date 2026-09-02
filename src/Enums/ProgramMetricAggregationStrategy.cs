namespace AiDotNet.Enums;

/// <summary>Selects the rule that turns a dictionary of evaluator metrics into one scalar quality.</summary>
/// <remarks>
/// <para>
/// An evaluator usually reports several numbers - accuracy, latency, memory, a penalty - but a quality-diversity
/// archive ranks candidates on exactly one. This enumeration names the rule used to collapse the dictionary, and
/// naming it is the point: an aggregation that is chosen explicitly can be validated, logged, and reproduced,
/// whereas an aggregation that happens implicitly inside an evaluator cannot.
/// </para>
/// <para>
/// <see cref="CombinedScoreOrMean"/> reproduces the reference OpenEvolve rule
/// (<c>openevolve/utils/metrics_utils.py</c> <c>get_fitness_score</c>): prefer a metric literally named
/// <c>combined_score</c>, and otherwise average the numeric metrics that are not archive feature dimensions.
/// The remaining members are deliberate alternatives with no upstream counterpart.
/// </para>
/// <para><b>For Beginners:</b> Your scoring code hands back several measurements, but the search needs a single
/// number to decide which program is better. This setting says how those measurements are combined. Start with
/// <see cref="CombinedScoreOrMean"/>: have your evaluator report a metric called <c>combined_score</c> and that
/// value is used directly. Use <see cref="Weighted"/> when you know how much each measurement matters, and
/// <see cref="Tchebycheff"/> when you want the search to improve whichever measurement is currently furthest from
/// your target rather than letting a strong measurement hide a weak one.</para>
/// </remarks>
public enum ProgramMetricAggregationStrategy
{
    /// <summary>Uses a metric named <c>combined_score</c> when present, and otherwise the mean of numeric metrics.</summary>
    /// <remarks>The reference OpenEvolve rule, reproduced including its handling of flags, text, and missing values.</remarks>
    CombinedScoreOrMean = 0,

    /// <summary>Always averages the numeric metrics, ignoring any metric named <c>combined_score</c>.</summary>
    /// <remarks>The same averaging rule as <see cref="CombinedScoreOrMean"/> without the preferred-key shortcut.</remarks>
    Mean = 1,

    /// <summary>Combines metrics as a weighted mean using explicitly declared, validated weights.</summary>
    /// <remarks>Larger values are better. Metrics with no declared weight are reported rather than silently dropped.</remarks>
    Weighted = 2,

    /// <summary>Scores the largest weighted shortfall from a declared reference point.</summary>
    /// <remarks>
    /// Smaller values are better. This is the weighted Chebyshev achievement scalarizing function, optionally
    /// augmented by a small multiple of the summed shortfalls to exclude weakly efficient solutions.
    /// </remarks>
    Tchebycheff = 3
}
