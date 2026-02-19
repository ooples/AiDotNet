using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for LlamaOmni TTS model.</summary>
public class LlamaOmniOptions : CodecTtsOptions
{
    public LlamaOmniOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int FirstPacketLatencyMs { get; set; } = 200;
}
