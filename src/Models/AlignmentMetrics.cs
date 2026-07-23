namespace AiDotNet.Models;

/// <summary>
/// Contains metrics for evaluating AI alignment with human values.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class AlignmentMetrics<T>
{
    /// <summary>
    /// Gets or sets the helpfulness score (0-1).
    /// </summary>
    /// <remarks>
    /// Measures how helpful and informative the model's responses are.
    /// </remarks>
    public double HelpfulnessScore { get; set; }

    /// <summary>
    /// Gets or sets the harmlessness score (0-1).
    /// </summary>
    /// <remarks>
    /// Measures how safe the model is and whether it avoids harmful outputs.
    /// </remarks>
    public double HarmlessnessScore { get; set; }

    /// <summary>
    /// Gets or sets the honesty score (0-1).
    /// </summary>
    /// <remarks>
    /// Measures whether the model is truthful and doesn't make up information.
    /// </remarks>
    public double HonestyScore { get; set; }

    /// <summary>
    /// Gets or sets the overall alignment score (0-1).
    /// </summary>
    /// <remarks>
    /// Combines helpfulness, harmlessness, and honesty into a single metric.
    /// </remarks>
    public double OverallAlignmentScore { get; set; }

    /// <summary>
    /// Gets or sets the preference match rate.
    /// </summary>
    /// <remarks>
    /// Percentage of outputs that match human preferences.
    /// </remarks>
    public double PreferenceMatchRate { get; set; }

    /// <summary>
    /// Gets or sets the constitutional compliance score.
    /// </summary>
    /// <remarks>
    /// How well the model follows constitutional principles.
    /// </remarks>
    public double ConstitutionalComplianceScore { get; set; }

    /// <summary>
    /// Gets or sets additional alignment metrics.
    /// </summary>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}
