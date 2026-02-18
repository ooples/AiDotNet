using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>Options for CoMoSpeech TTS model.</summary>
public class CoMoSpeechOptions : EndToEndTtsOptions
{
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumDecoderLayers { get; set; } = 6;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
    public int FlowDim { get; set; } = 256;
    public int NumFlowLayers { get; set; } = 4;
}
