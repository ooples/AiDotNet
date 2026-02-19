using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for KaniTTS TTS model.</summary>
public class KaniTTSOptions : CodecTtsOptions
{
    public KaniTTSOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
