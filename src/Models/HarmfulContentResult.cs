namespace AiDotNet.Models;

/// <summary>
/// Result of harmful content identification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class HarmfulContentResult<T>
{
    /// <summary>
    /// Gets or sets whether harmful content was detected.
    /// </summary>
    public bool HarmfulContentDetected { get; set; }

    /// <summary>
    /// Gets or sets the overall harm score (0-1, higher means more harmful).
    /// </summary>
    public double HarmScore { get; set; }

    /// <summary>
    /// Gets or sets scores for each harmful content category.
    /// </summary>
    public Dictionary<string, double> CategoryScores { get; set; } = new();

    /// <summary>
    /// Gets or sets the primary harmful category detected.
    /// </summary>
    public string? PrimaryHarmCategory { get; set; }

    /// <summary>
    /// Gets or sets all harmful categories detected above threshold.
    /// </summary>
    public string[] DetectedCategories { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets detailed harmful content findings.
    /// </summary>
    public List<HarmfulContentFinding> Findings { get; set; } = new();

    /// <summary>
    /// Gets or sets the recommended action based on the harm level.
    /// </summary>
    public string RecommendedAction { get; set; } = "Allow";

    /// <summary>
    /// Gets or sets additional detection details.
    /// </summary>
    public Dictionary<string, object> Details { get; set; } = new();
}

/// <summary>
/// Represents a specific harmful content finding.
/// </summary>
public class HarmfulContentFinding
{
    /// <summary>
    /// Gets or sets the category of harmful content.
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the severity score (0-1).
    /// </summary>
    public double Severity { get; set; }

    /// <summary>
    /// Gets or sets the description of the harmful content.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location where the content was found.
    /// </summary>
    public int Location { get; set; }

    /// <summary>
    /// Gets or sets the excerpt of the harmful content.
    /// </summary>
    public string Excerpt { get; set; } = string.Empty;
}
