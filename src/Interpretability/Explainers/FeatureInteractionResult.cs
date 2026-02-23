using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Represents the result of a Feature Interaction analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FeatureInteractionResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the pairwise H-statistics ((featureI, featureJ) -> H value).
    /// </summary>
    public Dictionary<(int, int), T> PairwiseHStatistics { get; set; } = new();

    /// <summary>
    /// Gets or sets the overall H-statistics for each feature.
    /// </summary>
    public T[] OverallHStatistics { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets the interpretation of an H-statistic value.
    /// </summary>
    public static string InterpretHStatistic(double h)
    {
        if (h < 0.05) return "Negligible";
        if (h < 0.20) return "Weak";
        if (h < 0.50) return "Moderate";
        return "Strong";
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string> { "Feature Interaction Analysis (H-statistics):" };
        lines.Add("");

        // Overall interactions
        lines.Add("Overall Interactions:");
        for (int i = 0; i < FeatureNames.Length; i++)
        {
            double h = NumOps.ToDouble(OverallHStatistics[i]);
            lines.Add($"  {FeatureNames[i]}: H = {h:F4} ({InterpretHStatistic(h)})");
        }

        // Top pairwise interactions
        lines.Add("");
        lines.Add("Top Pairwise Interactions:");
        var topPairs = PairwiseHStatistics
            .OrderByDescending(kvp => NumOps.ToDouble(kvp.Value))
            .Take(10);

        foreach (var kvp in topPairs)
        {
            double h = NumOps.ToDouble(kvp.Value);
            lines.Add($"  {FeatureNames[kvp.Key.Item1]} x {FeatureNames[kvp.Key.Item2]}: H = {h:F4} ({InterpretHStatistic(h)})");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
