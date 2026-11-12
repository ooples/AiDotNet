using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// Configuration options for the wav2vec 2.0 self-supervised speech model.
/// </summary>
/// <remarks>
/// <para>
/// wav2vec 2.0 (Baevski et al., 2020, Meta) learns speech representations via contrastive
/// learning over quantized speech units. Pre-trained on 960 hours of LibriSpeech, it achieves
/// WER 1.8% on test-clean with only 10 minutes of labeled data when fine-tuned for ASR.
/// </para>
/// <para>
/// <b>For Beginners:</b> wav2vec 2.0 pioneered self-supervised learning for speech. It
/// listens to raw audio, masks parts of it, and learns to predict the missing parts. This
/// teaches it a deep understanding of speech that can then be used for tasks like
/// transcription with very little labeled data.
/// </para>
/// </remarks>
public class Wav2Vec2Options : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("base" or "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the transformer hidden dimension.</summary>
    public int HiddenDim { get; set; } = 768;

    /// <summary>Gets or sets the number of transformer layers.</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Gets or sets the feed-forward dimension.</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Gets or sets the CNN feature encoder output dimension.</summary>
    public int FeatureEncoderDim { get; set; } = 512;

    /// <summary>Gets or sets whether the network is the LARGE-LV60 form: layer norm in every feature-encoder layer and
    /// pre-norm ("stable layer norm") Transformer layers.</summary>
    public bool StableLayerNorm { get; set; }

    #endregion

    #region Pre-training

    /// <summary>Gets or sets the number of quantization codebooks, G (2).</summary>
    public int NumQuantizationGroups { get; set; } = 2;

    /// <summary>Gets or sets the number of entries per codebook, V (320).</summary>
    public int QuantizationCodebookSize { get; set; } = 320;

    /// <summary>Gets or sets the quantized vector's width, d (256; the codebook entries are d/G wide).</summary>
    public int CodevectorDim { get; set; } = 256;

    /// <summary>Gets or sets the width the context and quantized vectors are projected to (256 BASE, 768 LARGE).</summary>
    public int ProjectedCodevectorDim { get; set; } = 256;

    /// <summary>Gets or sets the number of distractors per masked step, K (100).</summary>
    public int NumNegatives { get; set; } = 100;

    /// <summary>Gets or sets the contrastive temperature κ (0.1).</summary>
    public double ContrastiveTemperature { get; set; } = 0.1;

    /// <summary>Gets or sets the diversity loss weight α (0.1).</summary>
    public double DiversityWeight { get; set; } = 0.1;

    /// <summary>Gets or sets the weight of the L2 penalty on the feature encoder's output (10).</summary>
    public double FeaturePenaltyWeight { get; set; } = 10.0;

    /// <summary>Gets or sets the factor on the feature encoder's gradients in pre-training (0.1).</summary>
    public double FeatureGradientScale { get; set; } = 0.1;

    /// <summary>Gets or sets the probability a frame starts a masked span (0.065) of <see cref="MaskLength"/> frames.</summary>
    public double MaskProbability { get; set; } = 0.065;

    /// <summary>Gets or sets the masked span length M (10).</summary>
    public int MaskLength { get; set; } = 10;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate used when no paper recipe applies.</summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>Gets or sets the Transformer dropout and the dropout at the feature encoder's output (0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the probability of skipping a Transformer layer in training (0.05 BASE, 0.2 LARGE).</summary>
    public double LayerDrop { get; set; } = 0.05;

    #endregion

    /// <summary>The network these options describe.</summary>
    internal Wav2Vec2Configuration ToConfiguration() => new()
    {
        ConvDims = Enumerable.Repeat(FeatureEncoderDim, 7).ToArray(),
        HiddenSize = HiddenDim,
        Layers = NumLayers,
        Heads = NumAttentionHeads,
        IntermediateSize = FeedForwardDim,
        LayerNormFeatureEncoder = StableLayerNorm,
        StableLayerNorm = StableLayerNorm,
        ConvBias = StableLayerNorm,
        FeatureProjectionDropout = DropoutRate,
        HiddenDropout = DropoutRate,
        AttentionDropout = DropoutRate,
        ActivationDropout = 0.0,
        LayerDrop = LayerDrop,
        MaskTimeProbability = MaskProbability,
        MaskTimeLength = MaskLength,
    };

    /// <summary>The pre-training objective these options describe.</summary>
    internal Wav2Vec2PretrainingConfiguration ToPretrainingConfiguration() => new()
    {
        Groups = NumQuantizationGroups,
        EntriesPerGroup = QuantizationCodebookSize,
        CodevectorDim = CodevectorDim,
        ProjectedDim = ProjectedCodevectorDim,
        Negatives = NumNegatives,
        LogitTemperature = ContrastiveTemperature,
        DiversityWeight = DiversityWeight,
        FeaturePenaltyWeight = FeaturePenaltyWeight,
        MaskProbability = MaskProbability,
        FeatureGradientScale = FeatureGradientScale,
    };
}
