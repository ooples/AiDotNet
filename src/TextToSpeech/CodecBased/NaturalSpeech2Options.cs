using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for NaturalSpeech2 TTS model.</summary>
public class NaturalSpeech2Options : EndToEndTtsOptions
{
    public int DiffusionDim { get; set; } = 256;
    public new int NumDiffusionSteps { get; set; } = 100;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumHeads { get; set; } = 4;
    public new double DropoutRate { get; set; } = 0.1;
}
