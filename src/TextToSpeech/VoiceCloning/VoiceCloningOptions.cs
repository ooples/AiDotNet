using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.VoiceCloning;
/// <summary>Base options for voice cloning TTS models.</summary>
public class VoiceCloningOptions : CodecTtsOptions
{
    public new int SpeakerEmbeddingDim { get; set; } = 256;
    public double MinReferenceDurationSec { get; set; } = 3.0;
    public new int NumEncoderLayers { get; set; } = 6;
    public new int NumHeads { get; set; } = 8;
    public new double DropoutRate { get; set; } = 0.1;
}
