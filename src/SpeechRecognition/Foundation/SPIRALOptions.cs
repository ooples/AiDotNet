using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Foundation;

/// <summary>Options for SPIRAL: self-supervised pre-training for speech recognition.</summary>
public class SPIRALOptions : ModelOptions
{
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 768;
    public int NumEncoderLayers { get; set; } = 12;
    public int NumAttentionHeads { get; set; } = 12;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 5000;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
