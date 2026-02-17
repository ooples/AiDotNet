namespace AiDotNet.Safety.Fairness;

/// <summary>
/// Configuration for bias and fairness detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure which types of bias to check for
/// and which demographic groups to analyze.
/// </para>
/// </remarks>
public class BiasConfig
{
    /// <summary>Protected attributes to analyze (e.g., "gender", "race", "age"). Null = all.</summary>
    public string[]? ProtectedAttributes { get; set; }

    /// <summary>Disparity threshold for flagging bias (0.0-1.0). Default: 0.2.</summary>
    public double? DisparityThreshold { get; set; }

    /// <summary>Whether to check for intersectional bias. Default: false.</summary>
    public bool? IntersectionalAnalysis { get; set; }

    /// <summary>Whether to detect stereotypical associations. Default: true.</summary>
    public bool? StereotypeDetection { get; set; }

    internal string[] EffectiveProtectedAttributes => ProtectedAttributes ?? new[] { "gender", "race", "age" };
    internal double EffectiveDisparityThreshold => DisparityThreshold ?? 0.2;
    internal bool EffectiveIntersectionalAnalysis => IntersectionalAnalysis ?? false;
    internal bool EffectiveStereotypeDetection => StereotypeDetection ?? true;
}
