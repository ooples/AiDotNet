using AiDotNet.Models.Options;
namespace AiDotNet.TextToSpeech.ProprietaryAPI;
/// <summary>Options for NVIDIARivaTTS TTS model.</summary>
public class NVIDIARivaTTSOptions : TtsModelOptions
{
    public new int NumEncoderLayers { get; set; } = 4;
    public new int NumDecoderLayers { get; set; } = 4;
    public new int NumHeads { get; set; } = 4;
    public new double DropoutRate { get; set; } = 0.1;
    public int EncoderDim { get; set; } = 256;
    public int DecoderDim { get; set; } = 256;
}
