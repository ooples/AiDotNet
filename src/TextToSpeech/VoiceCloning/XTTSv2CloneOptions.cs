namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for the voice-cloning facade over the canonical XTTS-v2 graph.</summary>
public class XTTSv2CloneOptions : XTTSv2Options
{
    /// <summary>Creates an instance with the default settings.</summary>
    public XTTSv2CloneOptions() => MinReferenceDurationSec = 6;

    /// <summary>Creates a deep copy.</summary>
    public XTTSv2CloneOptions(XTTSv2CloneOptions other) : base(other)
    {
        Validation.Guard.NotNull(other);
        MinReferenceDurationSec = other.MinReferenceDurationSec;
    }

    /// <summary>The six-second enrollment length used by the release and paper evaluation.</summary>
    public double MinReferenceDurationSec { get; set; }
}
