using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for CoMoSpeech TTS model.</summary>
public class CoMoSpeechOptions : EndToEndTtsOptions
{
    public CoMoSpeechOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int FlowDim { get; set; } = 256;
    public int NumFlowLayers { get; set; } = 4;
}
