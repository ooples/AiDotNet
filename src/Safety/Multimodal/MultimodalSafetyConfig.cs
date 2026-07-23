namespace AiDotNet.Safety.Multimodal;

/// <summary>
/// Configuration for multimodal safety modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure cross-modal safety checking,
/// including text-image alignment and cross-modal attack detection.
/// </para>
/// </remarks>
public class MultimodalSafetyConfig
{
    /// <summary>Mismatch threshold for flagging cross-modal inconsistencies (0.0-1.0). Default: 0.5.</summary>
    public double? MismatchThreshold { get; set; }

    /// <summary>Whether to check text-image alignment. Default: true.</summary>
    public bool? TextImageAlignment { get; set; }

    /// <summary>Whether to check for cross-modal attacks. Default: true.</summary>
    public bool? CrossModalAttackDetection { get; set; }

    internal double EffectiveMismatchThreshold => MismatchThreshold ?? 0.5;
    internal bool EffectiveTextImageAlignment => TextImageAlignment ?? true;
    internal bool EffectiveCrossModalAttackDetection => CrossModalAttackDetection ?? true;
}
