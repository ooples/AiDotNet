namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for AudioLM TTS model.</summary>
public class AudioLMOptions : CodecTtsOptions
{
    public AudioLMOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
