using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>Options for NaturalSpeech3 TTS model.</summary>
public class NaturalSpeech3Options : EndToEndTtsOptions
{
    public int DiffusionDim { get; set; } = 256;
    public new int NumDiffusionSteps { get; set; } = 100;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumHeads { get; set; } = 4;
    public new double DropoutRate { get; set; } = 0.1;
}
