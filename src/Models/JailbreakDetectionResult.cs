namespace AiDotNet.Models;

/// <summary>
/// Result of jailbreak attempt detection.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class JailbreakDetectionResult<T>
{
    /// <summary>
    /// Gets or sets whether a jailbreak attempt was detected.
    /// </summary>
    public bool JailbreakDetected { get; set; }

    /// <summary>
    /// Gets or sets the confidence score for the detection (0-1).
    /// </summary>
    public double ConfidenceScore { get; set; }

    /// <summary>
    /// Gets or sets the type of jailbreak technique detected.
    /// </summary>
    public string JailbreakType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the severity of the jailbreak attempt (0-1).
    /// </summary>
    public double Severity { get; set; }

    /// <summary>
    /// Gets or sets specific patterns or indicators that triggered detection.
    /// </summary>
    public List<JailbreakIndicator> Indicators { get; set; } = new();

    /// <summary>
    /// Gets or sets recommended mitigation actions.
    /// </summary>
    public string[] RecommendedActions { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets additional detection details.
    /// </summary>
    public Dictionary<string, object> DetectionDetails { get; set; } = new();
}

/// <summary>
/// Represents a specific jailbreak indicator.
/// </summary>
public class JailbreakIndicator
{
    /// <summary>
    /// Gets or sets the type of indicator.
    /// </summary>
    public string Type { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of what was detected.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score for this indicator (0-1).
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Gets or sets the location where the indicator was found.
    /// </summary>
    public int Location { get; set; }
}
