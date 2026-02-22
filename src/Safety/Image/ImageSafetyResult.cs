using AiDotNet.Enums;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Detailed result from image safety classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ImageSafetyResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class ImageSafetyResult
{
    /// <summary>Whether the image is safe overall.</summary>
    public bool IsSafe { get; init; }

    /// <summary>Per-category safety scores (0.0 = safe, 1.0 = maximum risk).</summary>
    public IReadOnlyDictionary<SafetyCategory, double> CategoryScores { get; init; } = new Dictionary<SafetyCategory, double>();

    /// <summary>The highest-risk category detected.</summary>
    public SafetyCategory HighestRiskCategory { get; init; }

    /// <summary>The highest risk score across all categories.</summary>
    public double HighestRiskScore { get; init; }
}
