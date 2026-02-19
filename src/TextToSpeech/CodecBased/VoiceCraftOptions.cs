namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for VoiceCraft TTS model.</summary>
public class VoiceCraftOptions : CodecTtsOptions
{
    public VoiceCraftOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
