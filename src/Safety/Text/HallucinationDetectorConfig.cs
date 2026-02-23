namespace AiDotNet.Safety.Text;

/// <summary>
/// Configuration for hallucination detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure hallucination detection. You can set
/// the threshold for how confident the detector must be before flagging content as
/// hallucinated, and choose whether to use reference-based or self-consistency methods.
/// </para>
/// </remarks>
public class HallucinationDetectorConfig
{
    /// <summary>Hallucination score threshold (0.0-1.0). Default: 0.5.</summary>
    public double? Threshold { get; set; }

    /// <summary>Number of samples for self-consistency checking. Default: 3.</summary>
    public int? ConsistencySamples { get; set; }

    /// <summary>Whether to extract and verify individual claims. Default: true.</summary>
    public bool? PerClaimVerification { get; set; }

    internal double EffectiveThreshold => Threshold ?? 0.5;
    internal int EffectiveConsistencySamples => ConsistencySamples ?? 3;
    internal bool EffectivePerClaimVerification => PerClaimVerification ?? true;
}
