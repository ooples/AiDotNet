namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Configuration for audio watermarking modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure audio watermarking including the
/// embedding strength and technique.
/// </para>
/// </remarks>
public class AudioWatermarkConfig
{
    /// <summary>Watermark embedding strength (0.0-1.0). Default: 0.5.</summary>
    public double? Strength { get; set; }

    /// <summary>Watermarking technique to use. Default: SpreadSpectrum.</summary>
    public AudioWatermarkType? Technique { get; set; }

    internal double EffectiveStrength => Strength ?? 0.5;
    internal AudioWatermarkType EffectiveTechnique => Technique ?? AudioWatermarkType.SpreadSpectrum;
}

/// <summary>
/// The type of audio watermarking technique to use.
/// </summary>
public enum AudioWatermarkType
{
    /// <summary>Spread-spectrum frequency domain embedding.</summary>
    SpreadSpectrum,
    /// <summary>AudioSeal-style localized watermarking.</summary>
    AudioSeal
}
