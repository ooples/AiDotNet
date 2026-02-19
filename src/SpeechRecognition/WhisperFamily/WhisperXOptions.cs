using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.WhisperFamily;

/// <summary>Options for WhisperX (Bain et al., 2023): Whisper + forced alignment + VAD + diarization.</summary>
public class WhisperXOptions : ModelOptions
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
    /// <summary>Minimum speech segment duration in seconds for VAD.</summary>
    public double VadMinSpeechDuration { get; set; } = 0.25;
    /// <summary>Minimum silence duration in seconds for VAD segmentation.</summary>
    public double VadMinSilenceDuration { get; set; } = 0.1;
    /// <summary>Whether to enable speaker diarization.</summary>
    public bool EnableDiarization { get; set; } = false;
}
