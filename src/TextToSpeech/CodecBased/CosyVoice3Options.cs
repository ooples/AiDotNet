namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for CosyVoice3 TTS model.</summary>
public class CosyVoice3Options : CodecTtsOptions
{
    public CosyVoice3Options()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
