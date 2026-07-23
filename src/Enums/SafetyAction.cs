namespace AiDotNet.Enums;

/// <summary>
/// Defines the action to take when a safety violation is detected.
/// </summary>
/// <remarks>
/// <para>
/// Safety actions determine what happens when content fails a safety check.
/// Actions are ordered by severity, from most permissive to most restrictive.
/// </para>
/// <para>
/// <b>For Beginners:</b> When the safety system finds something potentially harmful,
/// it needs to decide what to do. These are the possible responses:
/// - Allow: Let it through (useful for logging-only modes)
/// - Log: Let it through but record it for review
/// - Warn: Let it through but attach a warning
/// - Modify: Change the content to make it safe (e.g., redact PII)
/// - Block: Stop the content entirely
/// - Quarantine: Block and flag for human review
/// </para>
/// </remarks>
public enum SafetyAction
{
    /// <summary>
    /// Allow the content through without modification.
    /// </summary>
    Allow,

    /// <summary>
    /// Allow the content but log the safety finding for later review.
    /// </summary>
    Log,

    /// <summary>
    /// Allow the content but attach a warning to the result.
    /// </summary>
    Warn,

    /// <summary>
    /// Modify the content to remove or redact the unsafe portion.
    /// </summary>
    Modify,

    /// <summary>
    /// Block the content entirely and return an error or safe fallback.
    /// </summary>
    Block,

    /// <summary>
    /// Block the content and flag it for human review.
    /// </summary>
    Quarantine
}
