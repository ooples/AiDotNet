using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>Options for Whisper.cpp: optimized C++ inference for Whisper models.</summary>
public class WhisperCPPOptions : ModelOptions
{
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 12;
    public int NumAttentionHeads { get; set; } = 8;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 51865;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
