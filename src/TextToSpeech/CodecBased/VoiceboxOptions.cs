namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for Voicebox TTS model.</summary>
public class VoiceboxOptions : CodecTtsOptions
{
    public VoiceboxOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
