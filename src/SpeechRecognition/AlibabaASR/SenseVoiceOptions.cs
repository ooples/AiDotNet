using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for SenseVoice: multi-task speech understanding model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure SenseVoice-Small. Supported defaults follow
/// the released FunASR configuration: 512-wide SAN-M-style encoder, 50 blocks, 4 attention heads,
/// 2048-wide feed-forward layers, 25,055 tokens, 80 mel bands, and AdamW at 2e-5.</para>
/// </remarks>
public class SenseVoiceOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SenseVoiceOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SenseVoiceOptions(SenseVoiceOptions other)
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
        DecoderDim = other.DecoderDim;
        NumDecoderLayers = other.NumDecoderLayers;
        FeedForwardDim = other.FeedForwardDim;
        UseCifAlignment = other.UseCifAlignment;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 50;
    public int NumAttentionHeads { get; set; } = 4;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 25055;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public int DecoderDim { get; set; } = 512;
    public int NumDecoderLayers { get; set; } = 0;
    public int FeedForwardDim { get; set; } = 2048;
    public bool UseCifAlignment { get; set; } = false;
    public double LearningRate { get; set; } = 2e-5;
    public double WeightDecay { get; set; } = 0.01;
}
