namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for CosyVoiceClone voice cloning model.</summary>
public class CosyVoiceCloneOptions : VoiceCloningOptions
{
    public CosyVoiceCloneOptions()
    {
        MinReferenceDurationSec = 3.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
