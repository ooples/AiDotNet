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

        // INHERITED FROM ModelOptions, AND THEREFORE EASY TO MISS. Every declared property is copied
        // below; Seed is not declared here, so it was silently dropped and a copied configuration
        // produced a DIFFERENT model from the one it was copied from -- the failure mode that costs
        // the most to diagnose, because the two configurations compare equal on everything visible.
        Seed = other.Seed;

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

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the hidden width of the encoder.</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 50;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 4;
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 25055;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Gets or sets the hidden width of the decoder.</summary>
    public int DecoderDim { get; set; } = 512;
    /// <summary>Gets or sets the number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 0;
    /// <summary>Gets or sets the inner width of each feed-forward block.</summary>
    public int FeedForwardDim { get; set; } = 2048;
    /// <summary>Gets or sets whether a continuous integrate-and-fire layer aligns encoder frames to output tokens before the decoder.</summary>
    public bool UseCifAlignment { get; set; } = false;
    /// <summary>Gets or sets the learning rate of the default optimizer.</summary>
    public double LearningRate { get; set; } = 2e-5;
    /// <summary>Gets or sets the weight decay of the default optimizer.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
