using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>Options for Parakeet-TDT (NVIDIA NeMo, 2024): 1.1B Conformer with Token-and-Duration Transducer.</summary>
public class ParakeetTDTOptions : ModelOptions
{
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 1024;
    public int PredictionDim { get; set; } = 640;
    public int JointDim { get; set; } = 640;
    public int NumEncoderLayers { get; set; } = 24;
    public int NumAttentionHeads { get; set; } = 16;
    public int FeedForwardDim { get; set; } = 4096;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 1024;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    /// <summary>Maximum duration tokens (number of frames to skip per emission).</summary>
    public int MaxDurationTokens { get; set; } = 4;
}
