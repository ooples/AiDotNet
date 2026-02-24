namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for UniAudio TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UniAudio model. Default values follow the original paper settings.</para>
/// </remarks>
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
