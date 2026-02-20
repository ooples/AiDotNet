using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for SpeechGPT TTS model.</summary>
public class SpeechGPTOptions : CodecTtsOptions
{
    public SpeechGPTOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
