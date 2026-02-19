using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>Options for Canary-Qwen (NVIDIA NeMo, 2025): multilingual ASR + translation with Qwen-2.5 LLM decoder.</summary>
public class CanaryQwenOptions : ModelOptions
{
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 1024;
    public int DecoderDim { get; set; } = 1536;
    public int NumEncoderLayers { get; set; } = 24;
    public int NumDecoderLayers { get; set; } = 28;
    public int NumAttentionHeads { get; set; } = 16;
    public int NumMels { get; set; } = 128;
    public int VocabSize { get; set; } = 151936;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public string[] SupportedLanguagesList { get; set; } = new[] { "en", "de", "es", "fr", "it", "pt", "nl", "ja", "ko", "zh", "hi", "ar", "ru", "uk", "pl", "sv", "fi", "no", "da", "tr" };
}
