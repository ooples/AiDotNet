using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>Options for VoxtLM: one decoder-only language model over speech units and text.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the paper's VoxtLM-k50 small model (Maiti et al. 2024, §4, the 125M rows
/// of Table 3) as ESPnet's <c>egs2/voxtlm_v1</c> recipe trains it: speech becomes one of 50 k-means clusters of
/// HuBERT Base's sixth-layer features every 20 ms; text is one of 2,000 subword tokens; and a 12-layer Transformer
/// language model of width 768 (12 heads, feed-forward 3,072, no positional encoding, dropout 0.1) predicts both.</para>
/// </remarks>
public class VoxtLMOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public VoxtLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VoxtLMOptions(VoxtLMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is inherited from ModelOptions and easy to miss; a copy without it builds a different model.
        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumSpeechUnits = other.NumSpeechUnits;
        TextVocabSize = other.TextVocabSize;
        EmbeddingDim = other.EmbeddingDim;
        ModelDim = other.ModelDim;
        NumLayers = other.NumLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        DropoutRate = other.DropoutRate;
        LabelSmoothing = other.LabelSmoothing;
        HubertWidth = other.HubertWidth;
        HubertLayer = other.HubertLayer;
        HubertHeads = other.HubertHeads;
        HubertIntermediate = other.HubertIntermediate;
        HubertConvChannels = other.HubertConvChannels;
        HubertPositionalKernel = other.HubertPositionalKernel;
        HubertPositionalGroups = other.HubertPositionalGroups;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        WarmupSteps = other.WarmupSteps;
        MaxGradientNorm = other.MaxGradientNorm;
        MaxTextLength = other.MaxTextLength;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>The speech tokenizer's k-means clusters (50, VoxtLM-k50; the paper also trains 200 and 1,000).</summary>
    public int NumSpeechUnits { get; set; } = 50;
    /// <summary>The text subword tokens (2,000, the paper's BPE size for k = 50).</summary>
    public int TextVocabSize { get; set; } = 2000;
    /// <summary>The token embedding's width (768, the recipe's <c>embed_unit</c>).</summary>
    public int EmbeddingDim { get; set; } = 768;
    /// <summary>The Transformer's width (768, the small model's F).</summary>
    public int ModelDim { get; set; } = 768;
    /// <summary>The Transformer layers (12, the small model's L).</summary>
    public int NumLayers { get; set; } = 12;
    /// <summary>The attention heads (12, the small model's H).</summary>
    public int NumAttentionHeads { get; set; } = 12;
    /// <summary>The feed-forward width (3,072, the recipe's <c>unit</c>).</summary>
    public int FeedForwardDim { get; set; } = 3072;
    /// <summary>Dropout on the input layer, the residual branches, the attention weights and the feed-forward block (0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Label smoothing of the loss (0.1, the recipe's <c>lsm_weight</c>).</summary>
    public double LabelSmoothing { get; set; } = 0.1;
    /// <summary>HuBERT Base's width (768).</summary>
    public int HubertWidth { get; set; } = 768;
    /// <summary>The HuBERT layer clustered (6, the recipe's <c>kmeans_feature: HuBERT_base/6</c>).</summary>
    public int HubertLayer { get; set; } = 6;
    /// <summary>HuBERT Base's attention heads (12).</summary>
    public int HubertHeads { get; set; } = 12;
    /// <summary>HuBERT Base's feed-forward width (3,072).</summary>
    public int HubertIntermediate { get; set; } = 3072;
    /// <summary>The channels of HuBERT's convolutional feature encoder (512).</summary>
    public int HubertConvChannels { get; set; } = 512;
    /// <summary>The kernel of HuBERT's convolutional position embedding (128).</summary>
    public int HubertPositionalKernel { get; set; } = 128;
    /// <summary>The groups of HuBERT's convolutional position embedding (16).</summary>
    public int HubertPositionalGroups { get; set; } = 16;
    /// <summary>Adam's peak learning rate (1e-3, the recipe's), reached at the end of warm-up.</summary>
    public double LearningRate { get; set; } = 1e-3;
    /// <summary>Adam's L2 weight decay (1e-6).</summary>
    public double WeightDecay { get; set; } = 1e-6;
    /// <summary>The <c>warmuplr</c> schedule's warm-up steps (10,000).</summary>
    public int WarmupSteps { get; set; } = 10000;
    /// <summary>The gradient-norm clip (5).</summary>
    public double MaxGradientNorm { get; set; } = 5.0;
    /// <summary>The longest transcript generated, in tokens (150, the recipe's ASR <c>maxlen</c>).</summary>
    public int MaxTextLength { get; set; } = 150;
    /// <summary>Optional text for each text-token index; without it, index <c>t</c> decodes as code point <c>t</c>.</summary>
    public string[]? Vocabulary { get; set; }
    /// <summary>Gets or sets the path to the model weights.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"en"</c>.</summary>
    public string Language { get; set; } = "en";
}
