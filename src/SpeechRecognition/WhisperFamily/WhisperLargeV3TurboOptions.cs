using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.WhisperFamily;

/// <summary>Options for Whisper large-v3-turbo (OpenAI, 2024) with only 4 decoder layers for 6x faster inference.</summary>
public class WhisperLargeV3TurboOptions : ModelOptions
{
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 1280;
    public int DecoderDim { get; set; } = 1280;
    public int NumEncoderLayers { get; set; } = 32;
    public int NumDecoderLayers { get; set; } = 4;
    public int NumAttentionHeads { get; set; } = 20;
    public int FeedForwardDim { get; set; } = 5120;
    public int NumMels { get; set; } = 128;
    public int VocabSize { get; set; } = 51866;
    public int MaxTextLength { get; set; } = 448;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.0;
    public string Language { get; set; } = "en";
}
