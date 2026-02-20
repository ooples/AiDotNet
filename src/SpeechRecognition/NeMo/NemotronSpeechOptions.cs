using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>Options for Nemotron-Speech (NVIDIA, 2025): multi-task ASR model with Nemotron LLM backbone.</summary>
public class NemotronSpeechOptions : ModelOptions
{
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 60;
    public int EncoderDim { get; set; } = 1024;
    public int DecoderDim { get; set; } = 2048;
    public int NumEncoderLayers { get; set; } = 24;
    public int NumDecoderLayers { get; set; } = 32;
    public int NumAttentionHeads { get; set; } = 16;
    public int NumMels { get; set; } = 128;
    public int VocabSize { get; set; } = 65536;
    public int MaxTextLength { get; set; } = 1024;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
