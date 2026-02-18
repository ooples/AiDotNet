using AiDotNet.TextToSpeech.VoiceCloning;
namespace AiDotNet.TextToSpeech.VoiceCloning;
/// <summary>Options for SeedTTSClone voice cloning model.</summary>
public class SeedTTSCloneOptions : VoiceCloningOptions
{
    public new double MinReferenceDurationSec { get; set; } = 5.0;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumLLMLayers { get; set; } = 12;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
}
