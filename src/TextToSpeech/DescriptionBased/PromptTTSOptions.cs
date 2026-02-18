using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.DescriptionBased;
/// <summary>Options for PromptTTS description-based TTS model.</summary>
public class PromptTTSOptions : EndToEndTtsOptions
{
    public int PromptEncoderDim { get; set; } = 128;
    public int NumPromptLayers { get; set; } = 3;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumDecoderLayers { get; set; } = 4;
    public new int NumHeads { get; set; } = 2;
    public new double DropoutRate { get; set; } = 0.1;
}
