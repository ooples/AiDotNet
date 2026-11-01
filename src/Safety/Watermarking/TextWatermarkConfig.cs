namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Configuration for text watermarking modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure text watermarking settings including
/// the watermark strength and detection threshold.
/// </para>
/// </remarks>
public class TextWatermarkConfig
{
    /// <summary>Watermark embedding strength (0.0-1.0). Default: 0.5.</summary>
    public double? Strength { get; set; }

    /// <summary>Detection threshold for watermark presence (0.0-1.0). Default: 0.5.</summary>
    public double? DetectionThreshold { get; set; }

    /// <summary>Watermarking technique to use. Default: Sampling.</summary>
    public TextWatermarkType? Technique { get; set; }

    internal double EffectiveStrength => Strength ?? 0.5;
    internal double EffectiveDetectionThreshold => DetectionThreshold ?? 0.5;
    internal TextWatermarkType EffectiveTechnique => Technique ?? TextWatermarkType.Sampling;
}

/// <summary>
/// The type of text watermarking technique to use.
/// </summary>
public enum TextWatermarkType
{
    /// <summary>Generation-time green-list logit bias (Kirchenbauer et al. 2023); see <see cref="SamplingWatermarker{T}"/>.</summary>
    Sampling,
    /// <summary>Keyed synonym substitution in finished text; see <see cref="LexicalWatermarker{T}"/>.</summary>
    Lexical,
    /// <summary>Keyed contraction and expansion in finished text; see <see cref="SyntacticWatermarker{T}"/>.</summary>
    Syntactic
}
