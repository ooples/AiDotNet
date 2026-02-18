using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>Options for VoiceFlow TTS model.</summary>
public class VoiceFlowOptions : EndToEndTtsOptions
{
    public int FlowDim { get; set; } = 256;
    public new int NumFlowSteps { get; set; } = 2;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumHeads { get; set; } = 2;
    public new double DropoutRate { get; set; } = 0.1;
}
