namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for MetaVoice1B TTS model.</summary>
public class MetaVoice1BOptions : VoiceCloningOptions
{
    public MetaVoice1BOptions()
    {
        MinReferenceDurationSec = 3.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int EncoderDim { get; set; } = 512;
    public int DecoderDim { get; set; } = 1024;
}
