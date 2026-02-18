using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.StyleEmotion;
/// <summary>Options for OpenVoice TTS model.</summary>
public class OpenVoiceOptions : EndToEndTtsOptions
{
    public int SpeakerEmbeddingDim { get; set; } = 256;
    public int NumToneColorLayers { get; set; } = 3;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumDecoderLayers { get; set; } = 4;
    public new int NumHeads { get; set; } = 2;
    public new double DropoutRate { get; set; } = 0.1;
}
