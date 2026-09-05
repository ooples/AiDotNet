using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Metrics;

/// <summary>One metric the aggregation could not use, together with the reason it was set aside.</summary>
/// <remarks>
/// <para>
/// Scalarization is where a silent bug hides best. The reference OpenEvolve rule filters out strings, booleans and
/// <c>NaN</c> without recording that it did so, so an evaluator that reports <c>{"accuracy": "0.9"}</c> as text
/// instead of a number scores <c>0.0</c> and looks exactly like an evaluator whose program genuinely failed. An
/// issue is the record of one such filtered value: which metric, why, and a short human-readable note.
/// </para>
/// <para>
/// Issues are informational, not fatal. An aggregation still returns a value when it produces issues; the caller
/// decides whether a run should stop, whether the candidate should be reported as failed, or whether the note is
/// merely worth logging. The description never echoes an unbounded metric value, so an issue is always safe to log.
/// </para>
/// <para><b>For Beginners:</b> If your score comes out as zero and you cannot see why, read this list. It names
/// each measurement that was left out of the calculation and says whether it was a message rather than a number, a
/// yes/no flag, a missing entry, or something the configuration deliberately excluded.</para>
/// </remarks>
public sealed class ProgramMetricIssue
{
    /// <summary>The longest description carried by an issue, in characters.</summary>
    public const int MaxDescriptionLength = 256;

    /// <summary>Initializes an issue.</summary>
    /// <param name="metricName">The metric the issue concerns.</param>
    /// <param name="reason">Why the metric was set aside.</param>
    /// <param name="description">A short bounded note; longer text is truncated.</param>
    /// <exception cref="ArgumentNullException"><paramref name="metricName"/> or <paramref name="description"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="metricName"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="reason"/> is not a defined enumeration value.</exception>
    public ProgramMetricIssue(string metricName, ProgramMetricIssueReason reason, string description)
    {
        Guard.NotNullOrWhiteSpace(metricName);
        Guard.NotNull(description);
        if (!Enum.IsDefined(typeof(ProgramMetricIssueReason), reason)) throw new ArgumentOutOfRangeException(nameof(reason));
        MetricName = metricName.Trim();
        Reason = reason;
        Description = description.Length > MaxDescriptionLength
            ? description.Substring(0, MaxDescriptionLength)
            : description;
    }

    /// <summary>Gets the name of the metric the issue concerns.</summary>
    public string MetricName { get; }

    /// <summary>Gets why the metric was set aside.</summary>
    public ProgramMetricIssueReason Reason { get; }

    /// <summary>Gets a short bounded note describing the issue.</summary>
    public string Description { get; }

    /// <summary>Returns the metric name and reason.</summary>
    /// <returns>A short diagnostic label.</returns>
    public override string ToString() => MetricName + ": " + Reason.ToString();
}
