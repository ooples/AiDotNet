using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>CosyVoice release options plus the convenience voice-cloning API's reference-duration contract.</summary>
/// <remarks>
/// The neural defaults are inherited unchanged from <see cref="CosyVoiceOptions"/>. This task facade consumes the same
/// checkpoints and must therefore retain CosyVoice-300M's dimensions rather than defining a second set of defaults.
/// </remarks>
public class CosyVoiceCloneOptions : CosyVoiceOptions
{
    /// <summary>Creates options with the released CosyVoice-300M defaults.</summary>
    public CosyVoiceCloneOptions()
    {
    }

    /// <summary>Initializes an independent copy of an existing configuration.</summary>
    public CosyVoiceCloneOptions(CosyVoiceCloneOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        MinReferenceDurationSec = other.MinReferenceDurationSec;
    }

    /// <summary>Minimum prompt length accepted by <c>IVoiceCloner</c>, whose input has no explicit sample rate.</summary>
    public double MinReferenceDurationSec { get; set; } = 3.0;
}
