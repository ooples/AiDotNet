using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for Seed-ASR: ByteDance's large-scale multilingual ASR.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the native Seed-ASR approximation. The paper
/// reports a nearly 2B-parameter LUISE encoder and a greater-than-10B-parameter MoE LLM, but does not
/// publish a complete layer-level configuration or exact optimizer hyperparameters. The defaults here
/// are therefore practical library defaults, and every exposed architecture/training value is customizable.</para>
/// </remarks>
public class SeedASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SeedASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SeedASROptions(SeedASROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 24;
    public int NumAttentionHeads { get; set; } = 8;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 32000;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public double LearningRate { get; set; } = 1e-4;
    public double WeightDecay { get; set; } = 0.01;
}
