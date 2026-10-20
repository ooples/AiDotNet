using AiDotNet.Attributes;
using AiDotNet.Audio.SpeechRecognition;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Wav2Vec2 speech recognition models.
/// </summary>
[DimensionDivisibility(nameof(HiddenDim), nameof(NumHeads))]
[DimensionDivisibility(nameof(HiddenDim), nameof(PositionalConvGroups))]
public class Wav2Vec2ModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Wav2Vec2ModelOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// The defaults are wav2vec 2.0 BASE (Baevski et al. 2020, Table 6 / fairseq): 12 layers of width 768 with 12
    /// heads and a 3072-wide feed-forward over the 7-layer feature encoder.
    /// </para>
    /// </remarks>
    public Wav2Vec2ModelOptions()
    {
        Language = "en";
        SampleRate = 16000;
        MaxAudioLengthSeconds = 30;
        HiddenDim = 768;
        NumTransformerLayers = 12;
        NumHeads = 12;
        FfDim = 3072;
    }


    /// <summary>
    /// Gets or sets the language.
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the max audio length seconds.
    /// </summary>
    public int MaxAudioLengthSeconds { get; set; }

    /// <summary>
    /// Gets or sets the hidden dim.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num transformer layers.
    /// </summary>
    public int NumTransformerLayers { get; set; }

    /// <summary>
    /// Gets or sets the num heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the ff dim.
    /// </summary>
    public int FfDim { get; set; }

    /// <summary>
    /// Gets or sets the groups of the convolutional relative positional embedding (16, Baevski et al. 2020, §4.2).
    /// </summary>
    /// <remarks>The convolution splits <see cref="HiddenDim"/> into this many groups, so it must divide the width.</remarks>
    public int PositionalConvGroups { get; set; } = 16;

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
        if (FinalDropout < 0 || FinalDropout >= 1) throw new ArgumentException("FinalDropout must be in [0, 1).");
    }

    /// <summary>
    /// Gets or sets whether the feature encoder uses layer norm on every layer with pre-norm "stable layer norm"
    /// Transformer layers (wav2vec 2.0 LARGE-LV60) instead of group norm on the first layer with post-norm layers
    /// (BASE).
    /// </summary>
    public bool StableLayerNorm { get; set; }

    /// <summary>
    /// Gets or sets the dropout before the CTC head (transformers' <c>final_dropout</c>, 0.1).
    /// </summary>
    public double FinalDropout { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether fine-tuning leaves the convolutional feature encoder frozen (the paper: "the feature
    /// encoder is not trained during fine-tuning").
    /// </summary>
    public bool FreezeFeatureEncoder { get; set; } = true;

    /// <summary>Gets or sets the encoder dropout (hidden_dropout, 0.1).</summary>
    public double HiddenDropout { get; set; } = 0.1;

    /// <summary>Gets or sets the attention dropout (0.1).</summary>
    public double AttentionDropout { get; set; } = 0.1;

    /// <summary>Gets or sets the feed-forward activation dropout (0.1).</summary>
    public double ActivationDropout { get; set; } = 0.1;

    /// <summary>Gets or sets the probability of skipping a Transformer layer in training (LayerDrop, 0.1).</summary>
    public double LayerDrop { get; set; } = 0.1;

    /// <summary>Gets or sets the SpecAugment time-mask probability during fine-tuning (0.05).</summary>
    public double MaskTimeProbability { get; set; } = 0.05;

    /// <summary>The network these options describe.</summary>
    internal AiDotNet.Audio.Foundations.Wav2Vec2Configuration ToConfiguration() => new()
    {
        HiddenSize = HiddenDim,
        Layers = NumTransformerLayers,
        Heads = NumHeads,
        IntermediateSize = FfDim,
        PositionalConvGroups = PositionalConvGroups,
        LayerNormFeatureEncoder = StableLayerNorm,
        StableLayerNorm = StableLayerNorm,
        ConvBias = StableLayerNorm,
        HiddenDropout = HiddenDropout,
        AttentionDropout = AttentionDropout,
        ActivationDropout = ActivationDropout,
        LayerDrop = LayerDrop,
        MaskTimeProbability = MaskTimeProbability,
    };
}
