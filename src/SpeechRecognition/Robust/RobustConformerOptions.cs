using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Robust;

/// <summary>Options for Wang et al. 2021's noise-robust wav2vec 2.0 (<see cref="RobustConformer{T}"/>).</summary>
/// <remarks>
/// <para>The defaults are section 3.2's LibriSpeech model: wav2vec 2.0 BASE (12 Transformer blocks of width 768) as
/// released in <c>wav2vec_small.pt</c>, the reconstruction module of section 2.2 weighted by λ2 = 0.1 (Equation 4), and
/// section 3.3's CTC fine-tuning over characters.</para>
/// <para><b>For Beginners:</b> These options set the size of the speech encoder, the reconstruction module used only in
/// noisy pre-training, and the regularization used while fine-tuning on transcripts.</para>
/// </remarks>
public class RobustConformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public RobustConformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RobustConformerOptions(RobustConformerOptions other)
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
        ReconstructionHidden = other.ReconstructionHidden;
        ReconstructionWeight = other.ReconstructionWeight;
        DiversityWeight = other.DiversityWeight;
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
        Language = other.Language;
        Seed = other.Seed;
    }

    /// <summary>Audio sample rate in Hz. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Longest audio, in seconds, a transcription reads; longer input is truncated. Default: 30.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Transformer width (section 3.2: 768).</summary>
    public int EncoderDim { get; set; } = 768;

    /// <summary>Transformer blocks (section 3.2: 12).</summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>Attention heads per block: 12, as in the released <c>wav2vec_small.pt</c> the paper continues from
    /// (section 3.2's text says 8, which does not divide that checkpoint's attention).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Feed-forward width of each block (BASE: 3,072).</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Groups of the convolutional positional embedding (16).</summary>
    public int PositionalConvGroups { get; set; } = 16;

    /// <summary>LSTM units per direction of the reconstruction module's BLSTM layers; the paper does not state them, so
    /// the default makes the BLSTM output the encoder width (384 per direction).</summary>
    public int ReconstructionHidden { get; set; } = 384;

    /// <summary>λ2, the reconstruction loss's weight in Equation 4 (0.1).</summary>
    public double ReconstructionWeight { get; set; } = 0.1;

    /// <summary>λ1, the diversity loss's weight in Equation 4 (0.1).</summary>
    public double DiversityWeight { get; set; } = 0.1;

    /// <summary>CTC vocabulary size, blank (token 0) included; tokens above 0 decode as Unicode code points. Default: 128,
    /// the ASCII characters section 3.3's character predictions are drawn from.</summary>
    public int VocabSize { get; set; } = 128;

    /// <summary>Most tokens a transcription emits. Default: 512.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Path of an exported ONNX model, for the ONNX constructor.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options for the ONNX constructor.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Dropout on the Transformer's hidden states and attention weights. Default: 0.1.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Dropout inside each feed-forward block. Default: 0.0.</summary>
    public double ActivationDropout { get; set; } = 0.0;

    /// <summary>Dropout after the feature projection. Default: 0.1.</summary>
    public double FeatureProjectionDropout { get; set; } = 0.1;

    /// <summary>Dropout before the CTC head. Default: 0.0.</summary>
    public double FinalDropout { get; set; } = 0.0;

    /// <summary>Probability of skipping each Transformer block during training. Default: 0.1.</summary>
    public double LayerDrop { get; set; } = 0.1;

    /// <summary>Time-mask probability while fine-tuning. Default: 0.075.</summary>
    public double MaskTimeProbability { get; set; } = 0.075;

    /// <summary>Whether fine-tuning keeps the convolutional feature encoder frozen. Default: true.</summary>
    public bool FreezeFeatureEncoder { get; set; } = true;

    /// <summary>Language reported for a transcription when none is given. Default: "en".</summary>
    public string Language { get; set; } = "en";

    // wav2vec 2.0 BASE: group-normalized feature encoder, post-norm Transformer, convolutions without bias.
    internal Audio.Foundations.Wav2Vec2Configuration ToConfiguration() => new()
    {
        HiddenSize = EncoderDim,
        Layers = NumEncoderLayers,
        Heads = NumAttentionHeads,
        IntermediateSize = FeedForwardDim,
        PositionalConvGroups = PositionalConvGroups,
        LayerNormFeatureEncoder = false,
        StableLayerNorm = false,
        ConvBias = false,
        FeatureProjectionDropout = FeatureProjectionDropout,
        HiddenDropout = DropoutRate,
        AttentionDropout = DropoutRate,
        ActivationDropout = ActivationDropout,
        LayerDrop = LayerDrop,
        MaskTimeProbability = MaskTimeProbability,
    };

    internal Audio.Foundations.Wav2Vec2PretrainingConfiguration ToPretrainingConfiguration() => new() { DiversityWeight = DiversityWeight };
}
