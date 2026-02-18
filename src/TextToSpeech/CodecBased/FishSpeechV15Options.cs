using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for FishSpeechV15 TTS model.</summary>
public class FishSpeechV15Options : CodecTtsOptions
{
    public new int TextEncoderDim { get; set; } = 256;
    public new int LLMDim { get; set; } = 1024;
    public new int NumLLMLayers { get; set; } = 12;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
}
