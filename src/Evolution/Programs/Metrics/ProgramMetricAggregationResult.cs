using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Metrics;

/// <summary>The scalar quality produced from a metrics dictionary, with the evidence behind it.</summary>
/// <remarks>
/// <para>
/// <see cref="Value"/> is the number an evolution task reports as a candidate's quality. Everything else on this
/// type exists so the number can be defended: <see cref="Strategy"/> names the rule that produced it,
/// <see cref="UsedCombinedScore"/> says whether a single preferred metric short-circuited the calculation,
/// <see cref="ContributingMetrics"/> lists exactly which metrics were combined, and <see cref="Issues"/> lists the
/// metrics that were not, with a reason for each. The reference OpenEvolve rule returns only the number.
/// </para>
/// <para>
/// <see cref="HasFiniteValue"/> is the check a caller must make before handing <see cref="Value"/> to
/// <c>EvolutionTaskResult.Completed</c>, which requires a finite quality. It is <c>false</c> in exactly the cases
/// upstream would have propagated a <c>NaN</c> or an infinity into the archive: a <c>combined_score</c> that is
/// itself not a number, or an infinite metric reaching a mean. A caller that sees <c>false</c> should report the
/// candidate as failed rather than let a non-comparable score into the archive.
/// </para>
/// <para><b>For Beginners:</b> This is the answer to "how good was this program?", plus the working. Read
/// <see cref="Value"/> for the score, check <see cref="HasFiniteValue"/> before using it, and if the score
/// surprises you read <see cref="Issues"/> - it will tell you, for example, that the measurement you cared about
/// arrived as text and so was left out of the average.</para>
/// </remarks>
public sealed class ProgramMetricAggregationResult
{
    private readonly ReadOnlyCollection<ProgramMetricIssue> _issues;
    private readonly ReadOnlyCollection<string> _contributingMetrics;

    /// <summary>Initializes an aggregation result.</summary>
    /// <param name="value">The scalar quality, which may be non-finite when the inputs were.</param>
    /// <param name="strategy">The rule that produced <paramref name="value"/>.</param>
    /// <param name="usedCombinedScore">Whether a preferred combined-score metric short-circuited the calculation.</param>
    /// <param name="contributingMetrics">The names of the metrics that were combined, in ordinal order.</param>
    /// <param name="issues">The metrics that were set aside, with a reason each.</param>
    /// <exception cref="ArgumentNullException"><paramref name="contributingMetrics"/> or <paramref name="issues"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="strategy"/> is not a defined enumeration value.</exception>
    /// <exception cref="ArgumentException">A contributing metric name is empty, or an issue is <c>null</c>.</exception>
    public ProgramMetricAggregationResult(
        double value,
        ProgramMetricAggregationStrategy strategy,
        bool usedCombinedScore,
        IEnumerable<string> contributingMetrics,
        IEnumerable<ProgramMetricIssue> issues)
    {
        Guard.NotNull(contributingMetrics);
        Guard.NotNull(issues);
        if (!Enum.IsDefined(typeof(ProgramMetricAggregationStrategy), strategy))
            throw new ArgumentOutOfRangeException(nameof(strategy));

        string[] contributing = contributingMetrics.ToArray();
        if (contributing.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Contributing metric names cannot be empty.", nameof(contributingMetrics));
        ProgramMetricIssue[] issueCopy = issues.ToArray();
        if (issueCopy.Any(item => item is null))
            throw new ArgumentException("Issues cannot contain null entries.", nameof(issues));

        Value = value;
        Strategy = strategy;
        UsedCombinedScore = usedCombinedScore;
        _contributingMetrics = Array.AsReadOnly(contributing);
        _issues = Array.AsReadOnly(issueCopy);
    }

    /// <summary>Gets the scalar quality.</summary>
    public double Value { get; }

    /// <summary>Gets whether <see cref="Value"/> is a finite number and therefore usable as an archive quality.</summary>
    public bool HasFiniteValue => !double.IsNaN(Value) && !double.IsInfinity(Value);

    /// <summary>Gets the rule that produced <see cref="Value"/>.</summary>
    public ProgramMetricAggregationStrategy Strategy { get; }

    /// <summary>Gets whether a preferred combined-score metric short-circuited the calculation.</summary>
    public bool UsedCombinedScore { get; }

    /// <summary>Gets the names of the metrics that were combined, in ordinal order.</summary>
    public IReadOnlyList<string> ContributingMetrics => _contributingMetrics;

    /// <summary>Gets the metrics that were set aside, with a reason for each.</summary>
    public IReadOnlyList<ProgramMetricIssue> Issues => _issues;

    /// <summary>Returns the value, the strategy, and how many metrics were used and set aside.</summary>
    /// <returns>A short diagnostic label that never echoes metric content.</returns>
    public override string ToString() =>
        "ProgramMetricAggregationResult(" + Value.ToString("R", CultureInfo.InvariantCulture) +
        ", " + Strategy.ToString() +
        ", used=" + _contributingMetrics.Count.ToString(CultureInfo.InvariantCulture) +
        ", issues=" + _issues.Count.ToString(CultureInfo.InvariantCulture) + ")";
}
