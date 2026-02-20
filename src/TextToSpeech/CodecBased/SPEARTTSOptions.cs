namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for SPEARTTS TTS model.</summary>
public class SPEARTTSOptions : CodecTtsOptions
{
    public SPEARTTSOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
