namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for SeedTTSClone voice cloning model.</summary>
public class SeedTTSCloneOptions : VoiceCloningOptions
{
    public SeedTTSCloneOptions()
    {
        MinReferenceDurationSec = 5.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
