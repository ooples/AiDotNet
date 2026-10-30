using AiDotNet.Attributes;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Foundation;

/// <summary>Options for WavLM fine-tuned for speech recognition with CTC.</summary>
/// <remarks>
/// <para>WavLM (Chen et al. 2022) is wav2vec 2.0's convolutional feature encoder and Transformer with a gated relative position bias in every attention layer, pre-trained by masked prediction on denoised, overlapped speech and fine-tuned with a CTC head over raw 16 kHz audio. The defaults are the released microsoft/wavlm-base-plus configuration (group norm on the first feature-encoder layer, post-norm Transformer); set StableLayerNorm with 1024 wide, 24 layers, 16 heads and 4096 feed-forward for WavLM-Large.</para>
/// <para><b>For Beginners:</b> These options configure the WavLM recognizer. The model listens to the raw audio waveform (not a spectrogram) and outputs one character distribution per 20 ms frame.</para>
/// </remarks>
[DimensionDivisibility(nameof(EncoderDim), nameof(NumAttentionHeads))]
[DimensionDivisibility(nameof(EncoderDim), nameof(PositionalConvGroups))]
public class WavLMASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public WavLMASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WavLMASROptions(WavLMASROptions other)
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
        StableLayerNorm = other.StableLayerNorm;
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
    }

    /// <summary>Audio sample rate in Hz. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Longest audio, in seconds, a transcription reads; longer input is truncated. Default: 30.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Transformer width (hidden size). Default: 768 (WavLM base-plus).</summary>
    public int EncoderDim { get; set; } = 768;

    /// <summary>Number of Transformer layers. Default: 12.</summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>Attention heads per layer. Default: 12.</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Feed-forward width of each Transformer layer. Default: 3072.</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Groups of the convolutional positional embedding. Default: 16.</summary>
    public int PositionalConvGroups { get; set; } = 16;

    /// <summary>
    /// Whether to build the LARGE form: a layer-norm feature encoder with conv bias and a stable (pre-norm)
    /// Transformer. False builds the BASE form: group norm on the first feature-encoder layer and a post-norm
    /// Transformer. Default: false.
    /// </summary>
    public bool StableLayerNorm { get; set; }

    /// <summary>CTC vocabulary size, blank (token 0) included; tokens above 0 decode as Unicode code points. Default: 5000.</summary>
    public int VocabSize { get; set; } = 5000;

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

    /// <summary>Probability of skipping each Transformer layer during training. Default: 0.05.</summary>
    public double LayerDrop { get; set; } = 0.05;

    /// <summary>SpecAugment time-mask probability during training. Default: 0.05.</summary>
    public double MaskTimeProbability { get; set; } = 0.05;

    /// <summary>Whether fine-tuning keeps the convolutional feature encoder frozen. Default: true.</summary>
    public bool FreezeFeatureEncoder { get; set; } = true;

    /// <summary>Language reported for a transcription when none is given. Default: "en".</summary>
    public string Language { get; set; } = "en";

    // The WavLM network configuration these options describe (WavLM-Large's layout when StableLayerNorm is set).
    internal Audio.Foundations.WavLmConfiguration ToConfiguration() => new()
    {
        HiddenSize = EncoderDim,
        Layers = NumEncoderLayers,
        Heads = NumAttentionHeads,
        IntermediateSize = FeedForwardDim,
        PositionalConvGroups = PositionalConvGroups,
        LayerNormFeatureEncoder = StableLayerNorm,
        StableLayerNorm = StableLayerNorm,
        UseMaskEmbedding = true,
    };

    // Train-mode dropout, LayerDrop and SpecAugment time masking, drawn from the given generator.
    internal Audio.Foundations.WavLmRegularization ToRegularization(Random random)
        => new(random, DropoutRate, DropoutRate, ActivationDropout, FeatureProjectionDropout, LayerDrop, MaskTimeProbability);
}
