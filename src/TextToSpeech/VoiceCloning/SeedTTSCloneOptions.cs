using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for the voice-cloning facade over Seed-TTS.</summary>
public class SeedTTSCloneOptions : SeedTTSOptions
{
    /// <summary>Creates the facade options.</summary>
    public SeedTTSCloneOptions()
    {
        // The evaluation setup uses reference utterances between 3 and 20 seconds.
        MinReferenceDurationSec = 3.0;
    }

    /// <summary>Creates a copy.</summary>
    public SeedTTSCloneOptions(SeedTTSCloneOptions other)
        : base(other)
    {
        Validation.Guard.NotNull(other);
        MinReferenceDurationSec = other.MinReferenceDurationSec;
        ReferenceText = other.ReferenceText;
    }

    /// <summary>Minimum prompt duration; the paper's evaluation references span 3-20 seconds.</summary>
    public double MinReferenceDurationSec { get; set; }

    /// <summary>Optional transcript for the enrollment recording, enabling the full text-and-audio ICL prompt.</summary>
    public string ReferenceText { get; set; } = string.Empty;
}
