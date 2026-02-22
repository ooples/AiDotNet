using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for AudioPaLM TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the AudioPaLM model. Default values follow the original paper settings.</para>
/// </remarks>
public class AudioPaLMOptions : EndToEndTtsOptions
{
    public AudioPaLMOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
