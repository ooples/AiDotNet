namespace AiDotNet.Models;

/// <summary>
/// Result of safety validation for an input.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class SafetyValidationResult<T>
{
    /// <summary>
    /// Gets or sets whether the input passed validation.
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets or sets the safety score (0-1, higher is safer).
    /// </summary>
    public double SafetyScore { get; set; }

    /// <summary>
    /// Gets or sets the list of validation issues found.
    /// </summary>
    public List<ValidationIssue> Issues { get; set; } = new();

    /// <summary>
    /// Gets or sets whether a jailbreak attempt was detected.
    /// </summary>
    public bool JailbreakDetected { get; set; }

    /// <summary>
    /// Gets or sets harmful content categories detected.
    /// </summary>
    public string[] DetectedHarmCategories { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the sanitized/cleaned input (if applicable).
    /// </summary>
    public T[]? SanitizedInput { get; set; }

    /// <summary>
    /// Gets or sets additional validation details.
    /// </summary>
    public Dictionary<string, object> ValidationDetails { get; set; } = new();
}

/// <summary>
/// Represents a specific validation issue.
/// </summary>
public class ValidationIssue
{
    /// <summary>
    /// Gets or sets the severity level of the issue.
    /// </summary>
    public string Severity { get; set; } = "Medium";

    /// <summary>
    /// Gets or sets the issue type/category.
    /// </summary>
    public string Type { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the issue.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location/index where the issue was found.
    /// </summary>
    public int Location { get; set; }
}
