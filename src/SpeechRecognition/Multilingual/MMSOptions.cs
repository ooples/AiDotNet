using AiDotNet.Attributes;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Multilingual;

/// <summary>Options for MMS: speech recognition in 1,000+ languages.</summary>
/// <remarks>
/// <para>MMS (Pratap et al. 2023) is the wav2vec 2.0 architecture (stable-layer-norm form) at 1B parameters,
/// fine-tuned for CTC with a small attention adapter in every encoder layer that is swapped per language. The
/// defaults are the released <c>facebook/mms-1b-all</c> configuration.</para>
/// <para><b>For Beginners:</b> These options configure the MMS model. The model listens to the raw audio waveform
/// (not a spectrogram) and outputs one character distribution per 20 ms frame.</para>
/// </remarks>
[DimensionDivisibility(nameof(EncoderDim), nameof(NumAttentionHeads))]
[DimensionDivisibility(nameof(EncoderDim), nameof(PositionalConvGroups))]
public class MMSOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MMSOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MMSOptions(MMSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        PositionalConvGroups = other.PositionalConvGroups;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        ActivationDropout = other.ActivationDropout;
        FeatureProjectionDropout = other.FeatureProjectionDropout;
        FinalDropout = other.FinalDropout;
        LayerDrop = other.LayerDrop;
        MaskTimeProbability = other.MaskTimeProbability;
        FreezeFeatureEncoder = other.FreezeFeatureEncoder;
        AdapterDim = other.AdapterDim;
        Language = other.Language;
    }

    /// <summary>Audio sample rate in Hz. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Longest audio, in seconds, a transcription reads; longer input is truncated. Default: 30.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Transformer width (hidden size). Default: 1280 (MMS 1B).</summary>
    public int EncoderDim { get; set; } = 1280;

    /// <summary>Number of Transformer layers. Default: 48.</summary>
    public int NumEncoderLayers { get; set; } = 48;

    /// <summary>Attention heads per layer. Default: 16.</summary>
    public int NumAttentionHeads { get; set; } = 16;

    /// <summary>Feed-forward width of each Transformer layer. Default: 5120.</summary>
    public int FeedForwardDim { get; set; } = 5120;

    /// <summary>Groups of the convolutional positional embedding. Default: 16.</summary>
    public int PositionalConvGroups { get; set; } = 16;

    /// <summary>CTC vocabulary size, blank (token 0) included; tokens above 0 decode as Unicode code points. Default: 5000.</summary>
    public int VocabSize { get; set; } = 5000;

    /// <summary>Most tokens a transcription emits. Default: 512.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Path of an exported ONNX model, for the ONNX constructor.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options for the ONNX constructor.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Dropout on the Transformer's hidden states and attention weights. Default: 0.05.</summary>
    public double DropoutRate { get; set; } = 0.05;

    /// <summary>Dropout inside each feed-forward block. Default: 0.05.</summary>
    public double ActivationDropout { get; set; } = 0.05;

    /// <summary>Dropout after the feature projection. Default: 0.05.</summary>
    public double FeatureProjectionDropout { get; set; } = 0.05;

    /// <summary>Dropout before the CTC head. Default: 0.05.</summary>
    public double FinalDropout { get; set; } = 0.05;

    /// <summary>Probability of skipping each Transformer layer during training. Default: 0.05.</summary>
    public double LayerDrop { get; set; } = 0.05;

    /// <summary>SpecAugment time-mask probability during training. Default: 0.05.</summary>
    public double MaskTimeProbability { get; set; } = 0.05;

    /// <summary>Whether fine-tuning keeps the convolutional feature encoder frozen. Default: true.</summary>
    public bool FreezeFeatureEncoder { get; set; } = true;

    /// <summary>Width of the per-language attention adapter in each encoder layer. Default: 16.</summary>
    public int AdapterDim { get; set; } = 16;

    /// <summary>Language reported for a transcription when none is given. Default: "en".</summary>
    public string Language { get; set; } = "en";

    // The wav2vec 2.0 (stable layer norm) configuration with attention adapters these options describe.
    internal Audio.Foundations.Wav2Vec2Configuration ToConfiguration() => new()
    {
        HiddenSize = EncoderDim,
        Layers = NumEncoderLayers,
        Heads = NumAttentionHeads,
        IntermediateSize = FeedForwardDim,
        PositionalConvGroups = PositionalConvGroups,
        LayerNormFeatureEncoder = true,
        StableLayerNorm = true,
        ConvBias = true,
        FeatureProjectionDropout = FeatureProjectionDropout,
        HiddenDropout = DropoutRate,
        AttentionDropout = DropoutRate,
        ActivationDropout = ActivationDropout,
        LayerDrop = LayerDrop,
        MaskTimeProbability = MaskTimeProbability,
        AdapterAttentionDim = AdapterDim,
    };
}
