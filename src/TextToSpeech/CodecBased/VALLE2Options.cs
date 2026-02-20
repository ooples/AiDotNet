namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for VALLE2 TTS model.</summary>
public class VALLE2Options : CodecTtsOptions
{
    public VALLE2Options()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
