using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.MultiModal;
/// <summary>Options for AudioPaLM TTS model.</summary>
public class AudioPaLMOptions : EndToEndTtsOptions
{
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumDecoderLayers { get; set; } = 6;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
}
