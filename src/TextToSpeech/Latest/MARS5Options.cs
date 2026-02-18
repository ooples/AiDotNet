using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.Latest;
/// <summary>Options for MARS5 TTS model.</summary>
public class MARS5Options : CodecTtsOptions
{
    public new int TextEncoderDim { get; set; } = 256;
    public new int LLMDim { get; set; } = 1024;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumLLMLayers { get; set; } = 12;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
}
