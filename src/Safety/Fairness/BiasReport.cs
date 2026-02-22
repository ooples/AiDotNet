namespace AiDotNet.Safety.Fairness;

/// <summary>
/// Detailed report from bias and fairness evaluation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> BiasReport provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class BiasReport
{
    /// <summary>Whether any bias was detected above the threshold.</summary>
    public bool BiasDetected { get; init; }

    /// <summary>Overall bias score (0.0 = fair, 1.0 = severely biased).</summary>
    public double OverallBiasScore { get; init; }

    /// <summary>Per-group analysis results.</summary>
    public IReadOnlyList<GroupBiasResult> GroupResults { get; init; } = Array.Empty<GroupBiasResult>();

    /// <summary>Detected stereotypical associations.</summary>
    public IReadOnlyList<string> StereotypesDetected { get; init; } = Array.Empty<string>();
}

/// <summary>
/// Bias analysis result for a specific demographic group.
/// </summary>
public class GroupBiasResult
{
    /// <summary>The demographic group name.</summary>
    public string Group { get; init; } = string.Empty;

    /// <summary>The protected attribute category (e.g., "gender", "race").</summary>
    public string Attribute { get; init; } = string.Empty;

    /// <summary>Sentiment score for this group (-1.0 to 1.0).</summary>
    public double SentimentScore { get; init; }

    /// <summary>Disparity from the overall mean sentiment.</summary>
    public double Disparity { get; init; }
}
