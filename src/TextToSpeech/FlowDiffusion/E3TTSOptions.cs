using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>Options for E3TTS TTS model.</summary>
public class E3TTSOptions : EndToEndTtsOptions
{
    public int DiffusionDim { get; set; } = 256;
    public new int NumDiffusionSteps { get; set; } = 50;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumHeads { get; set; } = 4;
    public new double DropoutRate { get; set; } = 0.1;
}
