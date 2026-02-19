using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for NaturalSpeech TTS model.</summary>
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
