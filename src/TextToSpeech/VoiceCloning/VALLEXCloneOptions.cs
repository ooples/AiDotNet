namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for VALLEXClone voice cloning model.</summary>
public class VALLEXCloneOptions : VoiceCloningOptions
{
    public VALLEXCloneOptions()
    {
        MinReferenceDurationSec = 3.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
