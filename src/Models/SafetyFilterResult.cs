namespace AiDotNet.Models;

/// <summary>
/// Result of safety filtering on model output.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class SafetyFilterResult<T>
{
    /// <summary>
    /// Gets or sets whether the output passed filtering.
    /// </summary>
    public bool IsSafe { get; set; }

    /// <summary>
    /// Gets or sets the safety score (0-1, higher is safer).
    /// </summary>
    public double SafetyScore { get; set; }

    /// <summary>
    /// Gets or sets the filtered/sanitized output.
    /// </summary>
    public T[] FilteredOutput { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets whether the output was modified during filtering.
    /// </summary>
    public bool WasModified { get; set; }

    /// <summary>
    /// Gets or sets the harmful content detected in the output.
    /// </summary>
    public string[] DetectedHarmCategories { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets detailed filtering actions taken.
    /// </summary>
    public List<FilterAction> Actions { get; set; } = new();

    /// <summary>
    /// Gets or sets additional filtering details.
    /// </summary>
    public Dictionary<string, object> FilteringDetails { get; set; } = new();
}

/// <summary>
/// Represents a filtering action taken on the output.
/// </summary>
public class FilterAction
{
    /// <summary>
    /// Gets or sets the type of action taken.
    /// </summary>
    public string ActionType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the reason for the action.
    /// </summary>
    public string Reason { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location where the action was applied.
    /// </summary>
    public int Location { get; set; }

    /// <summary>
    /// Gets or sets what was removed or replaced.
    /// </summary>
    public string OriginalContent { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the replacement content (if applicable).
    /// </summary>
    public string? ReplacementContent { get; set; }
}
