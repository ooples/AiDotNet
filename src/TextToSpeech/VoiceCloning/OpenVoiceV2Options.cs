namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for OpenVoiceV2 TTS model.</summary>
public class OpenVoiceV2Options : VoiceCloningOptions
{
    public OpenVoiceV2Options()
    {
        MinReferenceDurationSec = 3.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int EncoderDim { get; set; } = 512;
    public int DecoderDim { get; set; } = 192;
}
