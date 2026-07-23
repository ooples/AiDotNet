namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for Voicebox TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Voicebox model. Default values follow the original paper settings.</para>
/// </remarks>
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
