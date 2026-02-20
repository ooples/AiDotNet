namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for UniAudio TTS model.</summary>
public class UniAudioOptions : CodecTtsOptions
{
    public UniAudioOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
