using AiDotNet.Enums;

namespace AiDotNet.Safety;

/// <summary>
/// Exception thrown when content fails a safety check and the configuration requires throwing.
/// </summary>
/// <remarks>
/// <para>
/// This exception is thrown when <see cref="SafetyConfig.ThrowOnUnsafeInput"/> or
/// <see cref="SafetyConfig.ThrowOnUnsafeOutput"/> is true and the safety pipeline
/// detects content that should be blocked.
/// </para>
/// <para>
/// <b>For Beginners:</b> When the safety system finds dangerous content and is configured
/// to throw exceptions, this is the exception you'll catch. It contains the full
/// <see cref="SafetyReport"/> with details about what was found.
/// </para>
/// </remarks>
public class SafetyViolationException : InvalidOperationException
{
    /// <summary>
    /// Gets the safety report containing all findings.
    /// </summary>
    public SafetyReport Report { get; }

    /// <summary>
    /// Gets whether this violation was on input (true) or output (false).
    /// </summary>
    public bool IsInputViolation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="SafetyViolationException"/> class.
    /// </summary>
    /// <param name="report">The safety report with findings.</param>
    /// <param name="isInputViolation">True if the violation was on input, false if on output.</param>
    public SafetyViolationException(SafetyReport report, bool isInputViolation)
        : base(BuildMessage(report ?? throw new ArgumentNullException(nameof(report)), isInputViolation))
    {
        Report = report;
        IsInputViolation = isInputViolation;
    }

    private static string BuildMessage(SafetyReport report, bool isInput)
    {
        var direction = isInput ? "Input" : "Output";
        var categories = report.DetectedCategories.Count > 0
            ? string.Join(", ", report.DetectedCategories)
            : "Unknown";
        return $"{direction} failed safety check. " +
               $"Severity: {report.HighestSeverity}, " +
               $"Action: {report.OverallAction}, " +
               $"Categories: [{categories}], " +
               $"Findings: {report.Findings.Count}";
    }
}
