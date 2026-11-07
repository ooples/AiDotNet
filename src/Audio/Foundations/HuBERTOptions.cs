using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// Configuration options for the HuBERT (Hidden-Unit BERT) self-supervised speech model.
/// </summary>
/// <remarks>
/// <para>
/// HuBERT (Hsu et al., 2021, Meta) learns speech representations by predicting masked
/// discrete speech units derived from clustering. It achieves strong performance on speech
/// recognition, speaker verification, and emotion detection when fine-tuned.
/// </para>
/// <para>
/// <b>For Beginners:</b> HuBERT learns to understand speech by listening to millions of
/// hours of audio. It predicts hidden "units" in speech (like phonemes) without any labels.
/// After pre-training, it can be fine-tuned for tasks like transcription or speaker ID.
/// </para>
/// </remarks>
public class HuBERTOptions : ModelOptions
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

    /// <summary>
    /// Whether to build HuBERT-Large's layout: a layer-norm feature encoder with conv bias and a stable (pre-norm)
    /// Transformer. False (the default) is HuBERT-Base's: group norm on the first feature-encoder layer and a
    /// post-norm Transformer.
    /// </summary>
    public bool StableLayerNorm { get; set; }

    /// <summary>Probability of skipping each Transformer layer during training (facebook/hubert-base-ls960: 0.1).</summary>
    public double LayerDrop { get; set; } = 0.1;

    /// <summary>SpecAugment time-mask probability during training (facebook/hubert-base-ls960: 0.05).</summary>
    public double MaskTimeProbability { get; set; } = 0.05;

    /// <summary>Dropout inside each feed-forward block (facebook/hubert-base-ls960: 0.1).</summary>
    public double ActivationDropout { get; set; } = 0.1;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    // The wav2vec 2.0 network HuBERT runs (Hsu et al. 2021, section 3), as transformers' HubertModel builds it.
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
        ActivationDropout = ActivationDropout,
        LayerDrop = LayerDrop,
        MaskTimeProbability = MaskTimeProbability,
    };
}
