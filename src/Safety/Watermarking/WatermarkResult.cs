namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Detailed result from watermark detection across any modality.
/// </summary>
/// <remarks>
/// <para>
/// A keyed detector reports the test statistic behind its confidence: <see cref="ZScore"/> is approximately standard
/// normal for content that does not carry the detector's watermark, and <see cref="PValue"/> is the probability of a
/// z at least this large by chance. <see cref="Confidence"/> maps z to [0, 1] as described in
/// <see cref="WatermarkStatistics.Confidence"/>.
/// </para>
/// <para><b>For Beginners:</b> a small p-value (say below 0.0001) means the content almost certainly carries the
/// watermark of the key used for detection; a p-value near 0.5 means there is no evidence either way.</para>
/// </remarks>
public class WatermarkResult
{
    /// <summary>Whether a watermark was detected.</summary>
    public bool WatermarkDetected { get; init; }

    /// <summary>Detection confidence (0.0 = no watermark, 1.0 = certain).</summary>
    public double Confidence { get; init; }

    /// <summary>The detection z-score; approximately standard normal for content without this key's watermark.</summary>
    public double ZScore { get; init; }

    /// <summary>The one-sided p-value of <see cref="ZScore"/>: the chance of so much evidence without the watermark.</summary>
    public double PValue { get; init; } = 1.0;

    /// <summary>The number of units the test scored (tokens, substitution sites or transform coefficients).</summary>
    public int ScoredUnits { get; init; }

    /// <summary>The modality of the content checked (text, image, audio).</summary>
    public string Modality { get; init; } = string.Empty;

    /// <summary>The detected watermark type, if identifiable.</summary>
    public string WatermarkType { get; init; } = string.Empty;
}
