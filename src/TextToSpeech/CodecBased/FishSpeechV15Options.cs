namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for FishSpeechV15 TTS model.</summary>
public class FishSpeechV15Options : CodecTtsOptions
{
    public FishSpeechV15Options()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
