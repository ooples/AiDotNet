using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Voice-cloning facade options for the native VALL-E X architecture.</summary>
public class VALLEXCloneOptions : VALLEXOptions
{
    /// <summary>Creates the paper-faithful VALL-E X configuration.</summary>
    public VALLEXCloneOptions()
    {
        MinReferenceDurationSec = 3.0;
    }

    /// <summary>Copies a facade configuration, including the full codec and Transformer options.</summary>
    public VALLEXCloneOptions(VALLEXCloneOptions other) : base(other)
    {
        Validation.Guard.NotNull(other);
        MinReferenceDurationSec = other.MinReferenceDurationSec;
        ReferenceText = other.ReferenceText;
    }

    /// <summary>Creates facade options from a VALL-E X architecture configuration.</summary>
    public VALLEXCloneOptions(VALLEXOptions other) : base(other)
    {
        MinReferenceDurationSec = 3.0;
    }

    /// <summary>Community checkpoint axes; does not select the original Microsoft paper configuration.</summary>
    public new static VALLEXCloneOptions CommunityCheckpoint() => new(VALLEXOptions.CommunityCheckpoint());

    /// <summary>The three-second enrollment length used by VALL-E and VALL-E X.</summary>
    public double MinReferenceDurationSec { get; set; }

    /// <summary>
    /// Transcript paired with the reference audio for the transcript-less <c>IVoiceCloner</c> convenience method.
    /// VALL-E X requires it in Eq. 3; use the overload accepting a transcript when it changes per call.
    /// </summary>
    public string ReferenceText { get; set; } = string.Empty;
}
