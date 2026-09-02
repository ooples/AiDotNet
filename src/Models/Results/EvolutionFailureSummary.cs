namespace AiDotNet.Models.Results;

/// <summary>One retained failure from an evolution run, reduced to a code and a message.</summary>
/// <remarks>
/// <para>
/// A candidate that throws, times out, or returns nothing usable becomes a diagnostic rather than a fault of the
/// run. A bounded number of those diagnostics is kept so a finished run can explain itself. This is the redacted
/// projection of one: the stable <see cref="Code"/> is safe to group and alert on, while <see cref="Message"/> is
/// prose meant for a human.
/// </para>
/// <para>
/// <see cref="IsRedacted"/> matters when the failure came from evaluating untrusted material — a generated program,
/// say. A message flagged this way has already had its risky content removed by the engine, but it still originated
/// outside your code, so treat it as data to display rather than as text to act on.
/// </para>
/// <para><b>For Beginners:</b> When a candidate fails, the run does not stop; it records why and carries on. This
/// is one of those records. If a run produced nothing useful, reading these is normally the fastest way to find out
/// that, for example, every candidate timed out or every one was rejected for missing a required measurement.</para>
/// </remarks>
public sealed class EvolutionFailureSummary
{
    /// <summary>Gets or sets the stable failure code, such as <c>evaluator_exception</c> or <c>descriptor_missing</c>.</summary>
    public string Code { get; set; } = string.Empty;

    /// <summary>Gets or sets the human-readable explanation.</summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>Gets or sets whether the message was sanitized because it came from untrusted material.</summary>
    public bool IsRedacted { get; set; }
}
