namespace AiDotNet.Safety.Audio;

/// <summary>
/// Configuration for voice protection (anti-cloning) modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure voice protection strength.
/// Higher strength makes voice cloning harder but may be more audible.
/// </para>
/// </remarks>
public class VoiceProtectorConfig
{
    /// <summary>Protection strength (0.0-1.0). Default: 0.3.</summary>
    public double? Strength { get; set; }

    /// <summary>Protection technique to use. Default: Perturbation.</summary>
    public VoiceProtectionType? Technique { get; set; }

    /// <summary>Target sample rate. Default: 16000 Hz.</summary>
    public int? SampleRate { get; set; }

    private static double Clamp(double value, double min, double max)
    {
        return Math.Max(min, Math.Min(max, value));
    }

    internal double EffectiveStrength
    {
        get
        {
            double value = Strength ?? 0.3;
            return Clamp(value, 0.0, 1.0);
        }
    }

    internal VoiceProtectionType EffectiveTechnique => Technique ?? VoiceProtectionType.Perturbation;

    internal int EffectiveSampleRate
    {
        get
        {
            int value = SampleRate ?? 16000;
            return value > 0 ? value : 16000;
        }
    }
}

/// <summary>
/// The type of voice protection technique to use.
/// </summary>
public enum VoiceProtectionType
{
    /// <summary>Add imperceptible perturbations (SPEC-style).</summary>
    Perturbation,
    /// <summary>Embed watermarks (AudioSeal-style).</summary>
    Watermark,
    /// <summary>Psychoacoustic masking (VocalCrypt-style).</summary>
    Masking
}
