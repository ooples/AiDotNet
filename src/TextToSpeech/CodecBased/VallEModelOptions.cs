using AiDotNet.Audio.Generation;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The native codec, Transformer and decoding configuration shared by the VALL-E family.</summary>
/// <remarks>Concrete configurations add their own prompt policies: VALL-E samples a duration, VALL-E X reads an
/// explicit second utterance, and VALL-E 2 samples its own duration range.</remarks>
/// <example><code>var options = new VALLEXOptions { HiddenDim = 1024, NumHeads = 16 };</code></example>
public abstract class VallEModelOptions : ModelOptions
{
    /// <summary>Creates the shared paper architecture defaults.</summary>
    protected VallEModelOptions() { }

    /// <summary>Copies the shared settings and independently copies the codec configuration.</summary>
    /// <param name="other">The source configuration.</param>
    protected VallEModelOptions(VallEModelOptions other) : base(other)
    {
        SampleRate = other.SampleRate; HopSize = other.HopSize; HiddenDim = other.HiddenDim;
        NumHeads = other.NumHeads; NumEncoderLayers = other.NumEncoderLayers; NumDecoderLayers = other.NumDecoderLayers;
        DropoutRate = other.DropoutRate; LearningRate = other.LearningRate; WeightDecay = other.WeightDecay;
        MaxTextLength = other.MaxTextLength; ModelPath = other.ModelPath; OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        VocabSize = other.VocabSize;
        FeedForwardDim = other.FeedForwardDim; TextTokens = other.TextTokens;
        NumCodebooks = other.NumCodebooks; CodebookSize = other.CodebookSize;
        WarmupSteps = other.WarmupSteps; TrainingSteps = other.TrainingSteps;
        Temperature = other.Temperature; TopK = other.TopK; MaxCodesPerTextToken = other.MaxCodesPerTextToken;
        Codec = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.Codec);
        SamplingSeed = other.SamplingSeed; PrependBos = other.PrependBos; StopOnMostLikelyEndToken = other.StopOnMostLikelyEndToken;
    }

    /// <summary>The waveform sample rate, matching the codec.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("SampleRate must match Codec.SampleRate; paired native codec tests cover 16/24 kHz encoding and decoding.")]
    public int SampleRate { get; set; } = 24000;
    /// <summary>The samples per codec frame, matching the codec strides.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("HopSize must equal the product of Codec.Ratios; paired native codec tests verify decoded lengths at hops 160/320.")]
    public int HopSize { get; set; } = 320;
    /// <summary>The hidden width of both Transformers.</summary>
    public int HiddenDim { get; set; } = 1024;
    /// <summary>The attention heads of both Transformers.</summary>
    public int NumHeads { get; set; } = 16;
    /// <summary>The NAR Transformer depth, equal to the AR depth.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("NumEncoderLayers must equal NumDecoderLayers; paired depth 1/2 tests train both native AR and NAR objectives.")]
    public int NumEncoderLayers { get; set; } = 12;
    /// <summary>The AR Transformer depth, equal to the NAR depth.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("NumDecoderLayers must equal NumEncoderLayers; paired depth 1/2 tests train both native AR and NAR objectives.")]
    public int NumDecoderLayers { get; set; } = 12;
    /// <summary>The Transformer training dropout.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>The peak AdamW learning rate.</summary>
    public double LearningRate { get; set; } = 5e-4;
    /// <summary>The AdamW weight decay.</summary>
    public double WeightDecay { get; set; } = 0.01;
    /// <summary>The phonemes retained by the frontend before framing.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>The exported ONNX graph path, when a graph is loaded.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Settings for the exported ONNX graph runtime.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    // Derived from the selected phone table, not a configurable vocabulary size.
    internal int VocabSize { get; set; }

    /// <summary>Prepend the beginning-of-audio token used by the community VALL-E-X checkpoint.</summary>
    public virtual bool PrependBos { get; set; }
    /// <summary>Stop when EOS is most likely, even if sampling selected another token.</summary>
    public virtual bool StopOnMostLikelyEndToken { get; set; } = true;
    /// <summary>The feed-forward width of both Transformers.</summary>
    public int FeedForwardDim { get; set; } = 4096;
    /// <summary>The phoneme embedding rows, sufficient for the selected inventory.</summary>
    [AiDotNet.Attributes.NotScaled("holds every phoneme id of the model's inventory (93 to 105)")]
    public int TextTokens { get; set; } = 512;
    /// <summary>The codec codebooks modeled by AR and NAR together.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("NumCodebooks must match the codec bandwidth and quantizer count; paired native codec tests encode/decode two and three codebooks.")]
    public int NumCodebooks { get; set; } = 8;
    /// <summary>The codes per codebook, matching the codec.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("CodebookSize must equal Codec.CodebookSize; paired native codec tests encode/decode vocabularies of 64/128 codes.")]
    [AiDotNet.Attributes.NotScaled("the codec's codes per codebook, which it must equal")]
    public int CodebookSize { get; set; } = 1024;
    /// <summary>The linear warmup duration in updates.</summary>
    public int WarmupSteps { get; set; } = 32000;
    /// <summary>The updates over which the learning rate decays linearly to zero.</summary>
    public int TrainingSteps { get; set; } = 800000;
    /// <summary>The AR sampling temperature.</summary>
    public virtual double Temperature { get; set; } = 1.0;
    /// <summary>The AR top-k cut, or zero to retain every code.</summary>
    public virtual int TopK { get; set; }
    /// <summary>The AR generation limit in codes per phoneme.</summary>
    public int MaxCodesPerTextToken { get; set; } = 16;
    /// <summary>The pretrained codec (EnCodec 24 kHz at 6 kbps by default).</summary>
    public EnCodecOptions Codec { get; set; } = VALLEOptions.CreatePaperCodec();
    /// <summary>The seed of training draws and sampling.</summary>
    public int SamplingSeed { get; set; }
}
