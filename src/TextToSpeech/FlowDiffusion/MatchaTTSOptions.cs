using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for MatchaTTS TTS model.</summary>
public class MatchaTTSOptions : EndToEndTtsOptions
{
    public MatchaTTSOptions()
    {
        NumFlowSteps = 4;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int FlowDim { get; set; } = 256;
}
