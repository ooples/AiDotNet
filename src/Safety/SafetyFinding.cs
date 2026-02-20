using AiDotNet.Enums;

namespace AiDotNet.Safety;

/// <summary>
/// Represents a single safety finding from a safety module evaluation.
/// </summary>
/// <remarks>
/// <para>
/// A safety finding captures one specific issue detected during safety analysis,
/// including what was found, how severe it is, where it was found, and what action
/// is recommended.
/// </para>
/// <para>
/// <b>For Beginners:</b> When the safety system checks your content, it may find
/// zero or more problems. Each problem is represented as a SafetyFinding with details
/// about what was detected and how serious it is.
/// </para>
/// </remarks>
public class SafetyFinding
{
    /// <summary>
    /// Gets the safety category of this finding.
    /// </summary>
    public SafetyCategory Category { get; init; }

    /// <summary>
    /// Gets the severity level of this finding.
    /// </summary>
    public SafetySeverity Severity { get; init; }

    /// <summary>
    /// Gets the confidence score for this finding (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// Higher values indicate greater confidence that the finding is correct.
    /// A score of 1.0 means absolute certainty; 0.5 means borderline.
    /// </remarks>
    public double Confidence { get; init; }

    /// <summary>
    /// Gets a human-readable description of the finding.
    /// </summary>
    public string Description { get; init; } = string.Empty;

    /// <summary>
    /// Gets the recommended action for this finding.
    /// </summary>
    public SafetyAction RecommendedAction { get; init; } = SafetyAction.Block;

    /// <summary>
    /// Gets the name of the safety module that produced this finding.
    /// </summary>
    public string SourceModule { get; init; } = string.Empty;

    /// <summary>
    /// Gets the start offset (character index or sample index) of the finding in the input.
    /// </summary>
    /// <remarks>
    /// -1 indicates the finding applies to the entire input rather than a specific span.
    /// </remarks>
    public int SpanStart { get; init; } = -1;

    /// <summary>
    /// Gets the end offset (exclusive) of the finding in the input.
    /// </summary>
    public int SpanEnd { get; init; } = -1;

    /// <summary>
    /// Gets the excerpt of content that triggered this finding.
    /// </summary>
    /// <remarks>
    /// May be empty if the finding applies to the overall content rather than a specific excerpt.
    /// </remarks>
    public string Excerpt { get; init; } = string.Empty;
}
