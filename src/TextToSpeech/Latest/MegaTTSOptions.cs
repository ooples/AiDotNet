using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for MegaTTS TTS model.</summary>
public class MegaTTSOptions : EndToEndTtsOptions
{
    public MegaTTSOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
