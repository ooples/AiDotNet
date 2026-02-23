namespace AiDotNet.Safety.Text;

/// <summary>
/// Configuration for toxicity detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure how strict the toxicity detector should be.
/// Higher thresholds mean only very toxic content is flagged; lower thresholds catch
/// more subtle toxicity but may have more false positives.
/// </para>
/// </remarks>
public class ToxicityDetectorConfig
{
    /// <summary>Toxicity score threshold (0.0-1.0). Default: 0.5.</summary>
    public double? Threshold { get; set; }

    /// <summary>Categories to detect. Null = all categories.</summary>
    public string[]? Categories { get; set; }

    /// <summary>Languages to support. Default: English only.</summary>
    public string[]? Languages { get; set; }

    /// <summary>Whether to include per-span toxicity scores. Default: false.</summary>
    public bool? IncludeSpanScores { get; set; }

    internal double EffectiveThreshold => Threshold ?? 0.5;
    internal string[] EffectiveLanguages => Languages ?? new[] { "en" };
}
