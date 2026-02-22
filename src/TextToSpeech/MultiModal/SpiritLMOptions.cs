using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for SpiritLM TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SpiritLM model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpiritLMOptions : CodecTtsOptions
{
    public SpiritLMOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
