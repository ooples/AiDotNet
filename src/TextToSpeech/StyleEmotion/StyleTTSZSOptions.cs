using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.StyleEmotion;
/// <summary>Options for StyleTTSZS TTS model.</summary>
public class StyleTTSZSOptions : EndToEndTtsOptions
{
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumDecoderLayers { get; set; } = 6;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
    public int StyleDim { get; set; } = 256;
    public int NumStyleLayers { get; set; } = 4;
}
