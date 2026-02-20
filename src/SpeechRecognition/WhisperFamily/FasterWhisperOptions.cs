using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.WhisperFamily;

/// <summary>Options for Faster-Whisper (SYSTRAN/CTranslate2, 2023): CTranslate2-optimized Whisper with int8 quantization.</summary>
public class FasterWhisperOptions : ModelOptions
{
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 1280;
    public int DecoderDim { get; set; } = 1280;
    public int NumEncoderLayers { get; set; } = 32;
    public int NumDecoderLayers { get; set; } = 32;
    public int NumAttentionHeads { get; set; } = 20;
    public int NumMels { get; set; } = 128;
    public int VocabSize { get; set; } = 51866;
    public int MaxTextLength { get; set; } = 448;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.0;
    public string Language { get; set; } = "en";
    /// <summary>Compute type for quantization (int8, float16, float32).</summary>
    public string ComputeType { get; set; } = "int8";
    /// <summary>Number of parallel transcription beams.</summary>
    public int BeamSize { get; set; } = 5;
}
