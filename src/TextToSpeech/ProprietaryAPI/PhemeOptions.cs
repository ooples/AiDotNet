using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.ProprietaryAPI;
/// <summary>Options for Pheme TTS model.</summary>
public class PhemeOptions : EndToEndTtsOptions
{
    public PhemeOptions() { NumFlowSteps = 0; EncoderDim = 256; DecoderDim = 256; }
    public new int NumEncoderLayers { get; set; } = 4;
    public new int NumDecoderLayers { get; set; } = 4;
    public new int NumHeads { get; set; } = 4;
    public new double DropoutRate { get; set; } = 0.1;
}
