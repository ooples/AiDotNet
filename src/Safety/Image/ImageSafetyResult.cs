namespace AiDotNet.Safety.Image;

/// <summary>
/// Detailed result from image safety classification.
/// </summary>
public class ImageSafetyResult
{
    /// <summary>Whether the image is safe overall.</summary>
    public bool IsSafe { get; init; }

    /// <summary>Per-category safety scores (0.0 = safe, 1.0 = maximum risk).</summary>
    public IReadOnlyDictionary<string, double> CategoryScores { get; init; } = new Dictionary<string, double>();

    /// <summary>The highest-risk category detected.</summary>
    public string HighestRiskCategory { get; init; } = string.Empty;

    /// <summary>The highest risk score across all categories.</summary>
    public double HighestRiskScore { get; init; }
}
