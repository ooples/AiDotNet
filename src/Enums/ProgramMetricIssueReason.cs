namespace AiDotNet.Enums;

/// <summary>Explains why one metric did not contribute to a scalar quality in the way a caller might expect.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve scalarization silently discards every value it cannot use: a string metric, a boolean
/// flag, a <c>NaN</c>, or a metric excluded because it is an archive feature dimension all vanish without trace,
/// and an evaluator whose numbers never reach the fitness score looks identical to one whose numbers did. Each
/// member of this enumeration names one of those discards so the aggregation can report it instead.
/// </para>
/// <para><b>For Beginners:</b> When several measurements are combined into a single score, some of them may be
/// unusable - a message instead of a number, a yes/no flag, or a value that is missing. Rather than quietly
/// ignoring those, the aggregation returns a list of them with one of these reasons attached, so you can see at a
/// glance why your score is not what you expected.</para>
/// </remarks>
public enum ProgramMetricIssueReason
{
    /// <summary>A text metric was skipped because it could not be read as a finite number.</summary>
    NonNumericText = 0,

    /// <summary>A boolean flag was skipped because a flag is not a score.</summary>
    BooleanFlag = 1,

    /// <summary>A numeric metric was skipped because its value was not a number.</summary>
    NotANumber = 2,

    /// <summary>A numeric metric was infinite, so the aggregated value cannot be finite.</summary>
    NotFinite = 3,

    /// <summary>A metric named by the configuration was absent from the reported dictionary.</summary>
    MissingMetric = 4,

    /// <summary>A metric was excluded from the aggregation because it is an archive feature dimension.</summary>
    ExcludedFeatureDimension = 5,

    /// <summary>A numeric metric was reported but the weighted strategy declares no weight for it.</summary>
    NoWeightDeclared = 6,

    /// <summary>No usable numeric metric was found at all, so the aggregation fell back to zero.</summary>
    NoNumericValues = 7
}
