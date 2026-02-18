using AiDotNet.TextToSpeech.VoiceCloning;
namespace AiDotNet.TextToSpeech.VoiceCloning;
/// <summary>Options for VALLEXClone voice cloning model.</summary>
public class VALLEXCloneOptions : VoiceCloningOptions
{
    public new double MinReferenceDurationSec { get; set; } = 3.0;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumLLMLayers { get; set; } = 12;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
}
