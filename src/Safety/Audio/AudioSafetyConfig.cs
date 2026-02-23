namespace AiDotNet.Safety.Audio;

/// <summary>
/// Configuration for audio safety detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure audio safety detection including
/// deepfake detection, toxic speech detection, and voice protection settings.
/// </para>
/// </remarks>
public class AudioSafetyConfig
{
    /// <summary>Default sample rate for audio processing. Default: 16000 Hz.</summary>
    public int? SampleRate { get; set; }

    /// <summary>Deepfake detection threshold (0.0-1.0). Default: 0.5.</summary>
    public double? DeepfakeThreshold { get; set; }

    /// <summary>Toxicity detection threshold (0.0-1.0). Default: 0.5.</summary>
    public double? ToxicityThreshold { get; set; }

    /// <summary>Whether to enable voice protection (anti-cloning). Default: false.</summary>
    public bool? VoiceProtection { get; set; }

    internal int EffectiveSampleRate => SampleRate ?? 16000;
    internal double EffectiveDeepfakeThreshold => DeepfakeThreshold ?? 0.5;
    internal double EffectiveToxicityThreshold => ToxicityThreshold ?? 0.5;
    internal bool EffectiveVoiceProtection => VoiceProtection ?? false;
}
