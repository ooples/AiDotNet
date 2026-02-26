namespace AiDotNet.Enums;

/// <summary>
/// Indicates the severity level of a safety finding.
/// </summary>
/// <remarks>
/// <para>
/// Severity levels help prioritize safety findings and determine the appropriate response.
/// They follow standard security severity conventions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Not all safety issues are equally serious. A mild profanity
/// is different from CSAM or weapons instructions. Severity levels help you set
/// different thresholds and actions for different levels of risk.
/// </para>
/// </remarks>
public enum SafetySeverity
{
    /// <summary>
    /// Informational finding with no safety risk. Used for audit logging.
    /// </summary>
    Info,

    /// <summary>
    /// Low severity — content is borderline or slightly inappropriate.
    /// </summary>
    Low,

    /// <summary>
    /// Medium severity — content contains moderate safety concerns.
    /// </summary>
    Medium,

    /// <summary>
    /// High severity — content is clearly harmful or dangerous.
    /// </summary>
    High,

    /// <summary>
    /// Critical severity — content involves illegal activity, CSAM, or imminent danger.
    /// Must always be blocked regardless of configuration.
    /// </summary>
    Critical
}
