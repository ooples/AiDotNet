using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for NaturalSpeech TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the NaturalSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class NaturalSpeechOptions : EndToEndTtsOptions
{
    public NaturalSpeechOptions()
    {
        InterChannels = 192;
        FilterChannels = 768;
        NumFlowSteps = 4;
        NumEncoderLayers = 6;
        NumDecoderLayers = 8;
        NumHeads = 2;
        DropoutRate = 0.1;
    }
}
