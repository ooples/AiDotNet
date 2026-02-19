using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for MegaTTS2 TTS model.</summary>
public class MegaTTS2Options : EndToEndTtsOptions
{
    public MegaTTS2Options()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
