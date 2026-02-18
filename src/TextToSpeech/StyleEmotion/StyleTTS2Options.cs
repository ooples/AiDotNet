using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.StyleEmotion;
/// <summary>Options for StyleTTS2 TTS model.</summary>
public class StyleTTS2Options : EndToEndTtsOptions
{
    public int StyleDim { get; set; } = 128;
    public int NumStyleDiffusionSteps { get; set; } = 5;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumDecoderLayers { get; set; } = 4;
    public new int NumHeads { get; set; } = 2;
    public new double DropoutRate { get; set; } = 0.1;
}
