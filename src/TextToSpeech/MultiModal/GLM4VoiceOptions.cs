using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.MultiModal;
/// <summary>Options for GLM4Voice TTS model.</summary>
public class GLM4VoiceOptions : CodecTtsOptions
{
    public new int TextEncoderDim { get; set; } = 256;
    public new int LLMDim { get; set; } = 1024;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumLLMLayers { get; set; } = 12;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
    public int FirstPacketLatencyMs { get; set; } = 200;
}
