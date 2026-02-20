using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for VoiceFlow TTS model.</summary>
public class VoiceFlowOptions : EndToEndTtsOptions
{
    public VoiceFlowOptions()
    {
        NumFlowSteps = 2;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int FlowDim { get; set; } = 256;
}
