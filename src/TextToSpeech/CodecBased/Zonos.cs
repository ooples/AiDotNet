using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Effects;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Zonos, Zyphra's text-to-speech model (2025): a prefix of conditioning tokens — the text's eSpeak phonemes, a speaker
/// embedding, and targets for emotion, pitch variation, speaking rate, bandwidth and language — followed by DAC's nine
/// codebooks, delayed one step each, predicted autoregressively by a Transformer or a hybrid of Mamba2 and attention layers,
/// under classifier-free guidance; DAC decodes the codes to 44.1 kHz audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Zyphra released code and weights with a blog post, not a paper (Zyphra/Zonos; <c>Zyphra/Zonos-v0.1-transformer</c>,
/// <c>Zyphra/Zonos-v0.1-hybrid</c>, <c>Zyphra/Zonos-v0.1-speaker-embedding</c>). The blog fixes the design: eSpeak
/// phonemization, a prefix of conditioning embeddings, DAC tokens in Parler-TTS's delay pattern merged into one embedding per
/// step and predicted by nine heads, a simple autoregressive objective. The release gives the rest: the conditioners, the
/// backbones, guidance against a text-only prefix, min-p sampling with a repetition penalty, and the end-of-audio countdown
/// over the delayed codebooks.</para>
/// <para>Voice cloning uses the release's speaker model (a ResNet-293 with attentive statistics pooling and an LDA to 128
/// dimensions) on the reference recording; see <see cref="SpeakerEmbedding"/>.</para>
/// <para><b>For Beginners:</b> Zonos reads text aloud in a voice you choose from a few seconds of someone speaking, and lets
/// you steer the emotion, pitch variation and speed.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new Zonos&lt;float&gt;(architecture, new ZonosOptions());
/// model.LoadReferenceWeights("Zonos-v0.1-transformer/model.safetensors");
/// model.Codec.LoadPretrainedWeights("dac_44khz/model.safetensors");
/// model.LoadSpeakerModel("Zonos-v0.1-speaker-embedding");
/// var speaker = model.SpeakerEmbedding(recording, 44100);
/// var audio = model.Generate("Hello, world!", speaker);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Beta release of Zonos-v0.1", "https://www.zyphra.com/post/beta-release-of-zonos-v0-1", Year = 2025, Authors = "Zyphra")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
    Source = "Zyphra published no training recipe or code; the blog says only that Zonos learns a simple autoregressive "
             + "objective. Training uses the Hugging Face Trainer's defaults, as for the other released-code models here: AdamW "
             + "at 5e-5 with betas (0.9, 0.999), eps 1e-8, no weight decay, a linear decay to 0 without warm-up, gradient norm "
             + "clipped to 1.0.")]
public partial class Zonos<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly ZonosOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private ZonosNetwork<T>? _network;
    private ZonosPrefixConditioner<T>? _conditioner;
    private AudioCodecLayer<T>? _codec;
    private ZonosSpeakerEncoder<T>? _speakerEncoder;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly List<LayerBase<T>> _modelLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The languages Zonos accepts (eSpeak codes); a language's id is its index.</summary>
    public static IReadOnlyList<string> Languages { get; } = new[]
    {
        "af", "am", "an", "ar", "as", "az", "ba", "bg", "bn", "bpy", "bs", "ca", "cmn",
        "cs", "cy", "da", "de", "el", "en-029", "en-gb", "en-gb-scotland", "en-gb-x-gbclan",
        "en-gb-x-gbcwmd", "en-gb-x-rp", "en-us", "eo", "es", "es-419", "et", "eu", "fa",
        "fa-latn", "fi", "fr-be", "fr-ch", "fr-fr", "ga", "gd", "gn", "grc", "gu", "hak",
        "hi", "hr", "ht", "hu", "hy", "hyw", "ia", "id", "is", "it", "ja", "jbo", "ka",
        "kk", "kl", "kn", "ko", "kok", "ku", "ky", "la", "lfn", "lt", "lv", "mi", "mk",
        "ml", "mr", "ms", "mt", "my", "nb", "nci", "ne", "nl", "om", "or", "pa", "pap",
        "pl", "pt", "pt-br", "py", "quc", "ro", "ru", "ru-lv", "sd", "shn", "si", "sk",
        "sl", "sq", "sr", "sv", "sw", "ta", "te", "tn", "tr", "tt", "ur", "uz", "vi",
        "vi-vn-x-central", "vi-vn-x-south", "yue",
    };

    /// <summary>Creates an ONNX-backed Zonos for inference.</summary>
    public Zonos(NeuralNetworkArchitecture<T> architecture, string modelPath, ZonosOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ZonosOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Zonos.</summary>
    public Zonos(NeuralNetworkArchitecture<T> architecture, ZonosOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ZonosOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.Codec is null) throw new ArgumentException("Zonos needs its codec's options.", nameof(o.Codec));
        if (o.EndToken != o.Codec.CodebookSize || o.MaskToken != o.Codec.CodebookSize + 1)
            throw new ArgumentException($"Zonos's end and mask tokens follow the codec's {o.Codec.CodebookSize} codes.");
        if (o.CfgScale == 1.0) throw new ArgumentOutOfRangeException(nameof(o.CfgScale), o.CfgScale, "The release generates with guidance only (a scale other than 1).");
        if (o.Temperature < 0) throw new ArgumentOutOfRangeException(nameof(o.Temperature), o.Temperature, "The temperature is 0 (greedy) or positive.");
        if (o.ConditionerDropout < 0 || o.ConditionerDropout >= 1) throw new ArgumentOutOfRangeException(nameof(o.ConditionerDropout), o.ConditionerDropout, "Dropout lies in [0, 1).");
        if (!Languages.Contains(o.Language)) throw new ArgumentException($"'{o.Language}' is not one of Zonos's languages.", nameof(o.Language));
        if (o.Conditioners is null || o.Conditioners.Count == 0 || o.Conditioners[0].Kind != ZonosConditionerKind.EspeakPhonemes)
            throw new ArgumentException("Zonos's first conditioner reads the phonemes.", nameof(o.Conditioners));
        base.SampleRate = o.SampleRate;
        base.HopSize = o.Codec.EncoderRates.Aggregate(1, (a, b) => a * b);
        base.HiddenDim = o.HiddenDim;
        // Hugging Face's Trainer clips the gradient norm (max_grad_norm), which the optimizer does here.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.Codebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.Codec.CodebookSize;

    /// <inheritdoc />
    public int CodecFrameRate => (int)Math.Round(_options.SampleRate / (double)HopSize);

    /// <inheritdoc />
    /// <remarks>Zonos trains on a recording (44.1 kHz) or its DAC codes (<see cref="TtsTrainingSample{T}.CodecTokens"/>,
    /// <c>[frames, 9]</c>) with its text (<see cref="TtsTrainingSample{T}.Text"/>, or phoneme ids in
    /// <see cref="TtsTrainingSample{T}.Tokens"/>) and, when the speaker model is loaded, the speaker of its recording.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.Codebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.Codec.CodebookSize;

    private bool HasPaperLayers => _network is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal ZonosNetwork<T> Network => _network ?? throw NotNative();
    internal ZonosPrefixConditioner<T> Conditioner => _conditioner ?? throw NotNative();

    /// <summary>The DAC codec whose codes Zonos predicts (load <c>descript/dac_44khz</c> with
    /// <see cref="DAC{T}.LoadPretrainedWeights"/>).</summary>
    public DAC<T> Codec => (DAC<T>?)_codec?.Codec ?? throw NotNative();

    internal ZonosConfiguration NetworkConfiguration() => new()
    {
        Hidden = _options.HiddenDim, Layers = _options.Layers, AttentionLayers = _options.AttentionLayers, Heads = _options.Heads,
        KeyValueHeads = _options.KeyValueHeads, AttentionIntermediate = _options.AttentionIntermediate, MambaIntermediate = _options.MambaIntermediate,
        RotaryDim = _options.RotaryDim, RotaryInterleaved = _options.RotaryInterleaved, NormEpsilon = _options.NormEpsilon,
        StateDim = _options.StateDim, ConvWidth = _options.ConvWidth, Expand = _options.Expand, MambaHeadDim = _options.MambaHeadDim,
        Groups = _options.Groups, ChunkSize = _options.ChunkSize, Codebooks = _options.Codebooks,
        CodeVocab = _options.Codec.CodebookSize + 2, HeadVocab = _options.Codec.CodebookSize + 1,
    };

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(4 + ZonosPrefixConditioner<T>.Symbols.Length) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        _network = new ZonosNetwork<T>(Engine, _modelLayers, NetworkConfiguration());
        _conditioner = new ZonosPrefixConditioner<T>(Engine, _modelLayers, o.Conditioners, o.HiddenDim, random);
        InitializeLikeReference(random);

        // The codec (pretrained, frozen) only encodes prompts and training audio and decodes the output.
        var codecOptions = (DACOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        codecOptions.IncludeDiscriminators = false;
        var codec = new DAC<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
            { RandomSeed = Architecture.RandomSeed }, codecOptions);
        if (((IAudioCodec<T>)codec).SampleRate != o.SampleRate)
            throw new ArgumentException($"Zonos's {o.SampleRate} Hz audio must be its codec's ({((IAudioCodec<T>)codec).SampleRate} Hz).");
        if (codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps) != o.Codebooks)
            throw new ArgumentException($"Zonos predicts {o.Codebooks} codebooks; the codec's bandwidth gives {codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps)}.");
        _codec = new AudioCodecLayer<T>(codec);
        AddUnsplitLayers(_modelLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.Add(_codec);
    }

    // The release builds its modules with PyTorch's defaults: nn.Linear's kaiming-uniform weights and uniform biases
    // (±1/√fan-in), nn.Embedding N(0, 1), LayerNorm 1 and 0, the learned unconditional vectors 0, and mamba_ssm's Mamba2
    // initialization (dt_bias the inverse softplus of a log-uniform step in [0.001, 0.1], A_log the log of U(1, 16), D 1,
    // the gated norm's weight 1, the convolution PyTorch's default).
    private void InitializeLikeReference(Random random)
    {
        var c = Network.Configuration;
        double Uniform(double bound) => bound * (2 * random.NextDouble() - 1);
        double[] Fill(int count, Func<double> draw) => Enumerable.Range(0, count).Select(_ => draw()).ToArray();
        Network.LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            if (name.EndsWith("norm.weight", StringComparison.Ordinal) || name.EndsWith("norm2.weight", StringComparison.Ordinal)
                || name.EndsWith("norm_f.weight", StringComparison.Ordinal) || name.EndsWith(".D", StringComparison.Ordinal))
                return Enumerable.Repeat(1.0, count).ToArray();
            if (name.EndsWith("conv1d.bias", StringComparison.Ordinal)) return Fill(count, () => Uniform(1.0 / Math.Sqrt(c.ConvWidth)));
            if (name.EndsWith(".bias", StringComparison.Ordinal)) return new double[count];
            if (name.StartsWith("embeddings.", StringComparison.Ordinal)) return TorchParameters.Gaussian<double>(random, count, 1.0).ToArray();
            if (name.EndsWith("A_log", StringComparison.Ordinal)) return Fill(count, () => Math.Log(1 + 15 * random.NextDouble()));
            if (name.EndsWith("dt_bias", StringComparison.Ordinal))
                return Fill(count, () =>
                {
                    double dt = Math.Max(Math.Exp(random.NextDouble() * (Math.Log(0.1) - Math.Log(0.001)) + Math.Log(0.001)), 1e-4);
                    return dt + Math.Log(-(Math.Exp(-dt) - 1));
                });
            // Linear weights [out, in] and the depthwise convolution [channels, 1, width]: ±1/√fan-in.
            int fanIn = shape.Length == 3 ? shape[1] * shape[2] : shape[^1];
            return Fill(count, () => Uniform(1.0 / Math.Sqrt(fanIn)));
        });
        Conditioner.InitializeLikeReference(random);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Zonos-Native" : "Zonos-ONNX",
            Description = "Zonos: Zyphra's conditioned text-to-speech over delayed DAC codes with a Transformer or Mamba2 hybrid backbone (2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "Zonos";
        metadata.AdditionalInfo["Backbone"] = _options.Backbone.ToString();
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Zonos<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
