using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for SpeechT5 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SpeechT5 model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeechT5Options : EndToEndTtsOptions
{
    public SpeechT5Options()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
