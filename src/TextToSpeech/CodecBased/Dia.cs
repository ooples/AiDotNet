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
/// Dia, Nari Labs' dialogue model (2025): a Transformer encoder reads the script's UTF-8 bytes — speaker tags [S1] and [S2]
/// as bytes 1 and 2, non-verbal cues such as "(laughs)" as plain text — and an encoder-decoder Transformer predicts DAC's
/// nine codebooks at 86 frames a second, the codebooks delayed against each other, under classifier-free guidance; DAC
/// decodes the codes to 44.1 kHz audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Nari Labs released code and weights, not a paper (nari-labs/dia <c>dia/layers.py</c>, <c>dia/model.py</c>,
/// <c>dia/audio.py</c>; <c>nari-labs/Dia-1.6B-0626</c>). The decoder reads, at each step, the sum of the nine channels'
/// embeddings and predicts all nine; channel c runs <c>delay[c]</c> steps behind channel 0 ([0, 8, …, 15]), so a step's
/// codes are drawn knowing the earlier codebooks of the same frame. Generation runs the decoder twice per step, on the
/// script and on an all-zero script, keeps the codes whose guided logits <c>cond + s · (cond − uncond)</c> are among the
/// best k, and samples from the conditional logits among them with temperature and top-p; channel 0's end token starts a
/// countdown that closes each delayed channel with the end token and pads after it. An audio prompt's codes, with its
/// transcript at the front of the script, are prefilled to clone its voice.</para>
/// <para>Training follows transformers' DiaForConditionalGeneration, written with Nari Labs and the only training code:
/// the decoder reads the delayed codes between start and end tokens, and every channel's next token is scored by
/// cross-entropy, padding and start tokens excluded; the optimizer is the Trainer's default AdamW.</para>
/// <para><b>For Beginners:</b> Dia speaks a written dialogue. Mark who speaks with [S1] and [S2], add cues like
/// "(laughs)", and it voices both speakers in one recording. Give it a short recording and its transcript to make it
/// speak in that voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new Dia&lt;float&gt;(architecture, new DiaOptions());
/// model.LoadReferenceWeights("Dia-1.6B-0626/dia-v1.pth");
/// model.Codec.LoadDescriptCheckpoint("weights_44khz_8kbps_0.0.1.pth");
/// var audio = model.Synthesize("[S1] Dia is an open weights text to dialogue model. [S2] Wow. Amazing. (laughs)");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Dia: open-weights text-to-dialogue", "https://github.com/nari-labs/dia", Year = 2025, Authors = "Nari Labs")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
    Source = "Nari Labs published no training recipe and no training code. transformers' DiaForConditionalGeneration, written "
             + "with Nari Labs, trains through the Trainer, whose defaults are these: AdamW at 5e-5 with betas (0.9, 0.999), eps "
             + "1e-8, no weight decay, a linear decay to 0 without warm-up, gradient norm clipped to 1.0.")]
public partial class Dia<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly DiaOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private DiaNetwork<T>? _network;
    private AudioCodecLayer<T>? _codec;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly List<LayerBase<T>> _modelLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed Dia for inference.</summary>
    public Dia(NeuralNetworkArchitecture<T> architecture, string modelPath, DiaOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new DiaOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Dia.</summary>
    public Dia(NeuralNetworkArchitecture<T> architecture, DiaOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new DiaOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.Channels < 1 || o.DelayPattern is null || o.DelayPattern.Length != o.Channels)
            throw new ArgumentException("The delay pattern gives one delay per channel.", nameof(o.DelayPattern));
        if (o.DelayPattern[0] != 0 || o.DelayPattern.Any(d => d < 0))
            throw new ArgumentException("Channel 0 is not delayed and no delay is negative.", nameof(o.DelayPattern));
        if (o.Codec is null) throw new ArgumentException("Dia needs its codec's options.", nameof(o.Codec));
        int codes = o.Codec.CodebookSize;
        if (o.EndToken < codes || o.PadToken < codes || o.BeginToken < codes || new[] { o.EndToken, o.PadToken, o.BeginToken }.Distinct().Count() != 3
            || Math.Max(o.EndToken, Math.Max(o.PadToken, o.BeginToken)) >= o.AudioVocabSize)
            throw new ArgumentException($"The end, padding and start tokens are distinct, past the codec's {codes} codes and inside the {o.AudioVocabSize}-token tables.");
        if (o.CfgFilterTopK <= 0) throw new ArgumentOutOfRangeException(nameof(o.CfgFilterTopK), o.CfgFilterTopK, "The CFG filter keeps at least one code.");
        if (o.Temperature < 0) throw new ArgumentOutOfRangeException(nameof(o.Temperature), o.Temperature, "The temperature is 0 (greedy) or positive.");
        if (o.TopP <= 0 || o.TopP > 1) throw new ArgumentOutOfRangeException(nameof(o.TopP), o.TopP, "Top-p lies in (0, 1].");
        if (o.VocabSize < 256) throw new ArgumentException("Dia reads UTF-8 bytes; the text vocabulary has at least 256 entries.");
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
    public int NumCodebooks => _options.Channels;

    /// <inheritdoc />
    public int CodebookSize => _options.Codec.CodebookSize;

    /// <inheritdoc />
    public int CodecFrameRate => (int)Math.Round(_options.SampleRate / (double)HopSize);

    /// <inheritdoc />
    /// <remarks>Dia trains on a recording (44.1 kHz) or its DAC codes (<see cref="TtsTrainingSample{T}.CodecTokens"/>,
    /// <c>[frames, 9]</c> as <see cref="EncodeToTokens"/> gives them) with its script as byte ids
    /// (<see cref="TtsTrainingSample{T}.Tokens"/>) or <see cref="TtsTrainingSample{T}.Text"/>.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.Channels;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.Codec.CodebookSize;

    private bool HasPaperLayers => _network is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal DiaNetwork<T> Network => _network ?? throw NotNative();

    /// <summary>The DAC codec whose codes Dia predicts (load its released weights with
    /// <see cref="DAC{T}.LoadDescriptCheckpoint"/>).</summary>
    public DAC<T> Codec => (DAC<T>?)_codec?.Codec ?? throw NotNative();

    internal DiaConfiguration NetworkConfiguration() => new()
    {
        TextVocab = _options.VocabSize, EncoderHidden = _options.HiddenDim, EncoderLayers = _options.EncoderLayers,
        EncoderHeads = _options.EncoderHeads, EncoderKeyValueHeads = _options.EncoderKeyValueHeads, EncoderHeadDim = _options.EncoderHeadDim,
        EncoderIntermediate = _options.EncoderIntermediate, DecoderHidden = _options.DecoderHidden, DecoderLayers = _options.DecoderLayers,
        DecoderHeads = _options.DecoderHeads, DecoderKeyValueHeads = _options.DecoderKeyValueHeads, DecoderHeadDim = _options.DecoderHeadDim,
        CrossHeads = _options.CrossHeads, CrossKeyValueHeads = _options.CrossKeyValueHeads, CrossHeadDim = _options.CrossHeadDim,
        DecoderIntermediate = _options.DecoderIntermediate, AudioVocab = _options.AudioVocabSize, Channels = _options.Channels,
        NormEpsilon = _options.NormEpsilon, RopeTheta = _options.RopeTheta,
    };

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

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
        _network = new DiaNetwork<T>(Engine, _modelLayers, NetworkConfiguration());
        _network.InitializeLikeReference(random);

        // The codec (pretrained, frozen) only encodes prompts and decodes the output.
        var codecOptions = (DACOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        codecOptions.IncludeDiscriminators = false;
        var codec = new DAC<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
            { RandomSeed = Architecture.RandomSeed }, codecOptions);
        if (((IAudioCodec<T>)codec).SampleRate != o.SampleRate)
            throw new ArgumentException($"Dia's {o.SampleRate} Hz audio must be its codec's ({((IAudioCodec<T>)codec).SampleRate} Hz).");
        if (codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps) != o.Channels)
            throw new ArgumentException($"Dia predicts {o.Channels} codebooks; the codec's bandwidth gives {codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps)}.");
        _codec = new AudioCodecLayer<T>(codec);
        AddUnsplitLayers(_modelLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.Add(_codec);
    }

    // ---------------------------------------------------------------- text

    /// <summary>The script's byte ids (<c>Dia._encode_text</c>): its UTF-8 bytes with "[S1]" and "[S2]" as bytes 1 and 2,
    /// cut to <see cref="TtsModelOptions.MaxTextLength"/>.</summary>
    public static int[] Tokenize(string text, int maxLength = 1024)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var bytes = System.Text.Encoding.UTF8.GetBytes(text);
        var ids = new List<int>(bytes.Length);
        for (int i = 0; i < bytes.Length; i++)
        {
            if (i + 3 < bytes.Length && bytes[i] == (byte)'[' && bytes[i + 1] == (byte)'S' && (bytes[i + 2] == (byte)'1' || bytes[i + 2] == (byte)'2')
                && bytes[i + 3] == (byte)']')
            {
                ids.Add(bytes[i + 2] - '0');
                i += 3;
                continue;
            }
            ids.Add(bytes[i]);
        }
        return ids.Take(maxLength).ToArray();
    }

    private static string TextOf(Tensor<T> ids)
    {
        var bytes = new List<byte>();
        foreach (var v in ids.ToArray())
        {
            int id = (int)Math.Round(Ops.ToDouble(v));
            if (id == 1 || id == 2) bytes.AddRange(System.Text.Encoding.ASCII.GetBytes($"[S{id}]"));
            else if (id >= 0 && id < 256) bytes.Add((byte)id);
            else throw new ArgumentOutOfRangeException(nameof(ids), id, "Byte ids lie in [0, 256).");
        }
        return System.Text.Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <inheritdoc />
    /// <remarks>The script's byte ids.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Tokenize(text, _options.MaxTextLength);
        if (ids.Length == 0) throw new ArgumentException("The text has no bytes.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Dia-Native" : "Dia-ONNX",
            Description = "Dia: Nari Labs' text-to-dialogue encoder-decoder over delayed DAC codes with classifier-free guidance (2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "Dia";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Dia<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
