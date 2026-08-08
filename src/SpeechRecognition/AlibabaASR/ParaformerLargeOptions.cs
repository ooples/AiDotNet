using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for Paraformer-Large: 220M parameter version for production ASR.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ParaformerLarge model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class ParaformerLargeOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ParaformerLargeOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ParaformerLargeOptions(ParaformerLargeOptions other)
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
        DecoderDim = other.DecoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 1024;
    /// <summary>Gets or sets the parallel decoder width.</summary>
    /// <value>Defaults to 512, preserving the existing Paraformer-Large helper architecture.</value>
    public int DecoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 50;
    /// <summary>Gets or sets the number of non-autoregressive Transformer decoder blocks.</summary>
    /// <value>Defaults to 6, preserving the existing Paraformer-Large helper architecture.</value>
    public int NumDecoderLayers { get; set; } = 6;
    public int NumAttentionHeads { get; set; } = 16;
    /// <summary>Gets or sets the feed-forward inner width used by encoder and decoder blocks.</summary>
    /// <value>Defaults to 2048 hidden units.</value>
    public int FeedForwardDim { get; set; } = 2048;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 8404;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
