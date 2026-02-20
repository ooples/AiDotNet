namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for XTTSv2Clone voice cloning model.</summary>
public class XTTSv2CloneOptions : VoiceCloningOptions
{
    public XTTSv2CloneOptions()
    {
        MinReferenceDurationSec = 6.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
