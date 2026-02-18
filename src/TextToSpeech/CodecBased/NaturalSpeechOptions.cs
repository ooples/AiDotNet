using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for NaturalSpeech TTS model.</summary>
public class NaturalSpeechOptions : EndToEndTtsOptions
{
    public new int InterChannels { get; set; } = 192;
    public new int FilterChannels { get; set; } = 768;
    public new int NumFlowSteps { get; set; } = 4;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumDecoderLayers { get; set; } = 8;
    public new int NumHeads { get; set; } = 2;
    public new double DropoutRate { get; set; } = 0.1;
}
