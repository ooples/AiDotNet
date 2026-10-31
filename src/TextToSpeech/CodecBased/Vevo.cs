using System.Runtime.CompilerServices;
using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Resampling;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Vevo (Zhang et al., ICLR 2025): controllable zero-shot voice imitation by self-supervised disentanglement — HuBERT's
/// features quantized into content tokens (32 codes, timbre and style squeezed out) and content-style tokens (8,192 codes,
/// timbre squeezed out); an autoregressive transformer that turns content tokens (Vevo-Voice, Vevo-Style) or phones
/// (Vevo-TTS) into content-style tokens in a reference's style; and a flow-matching transformer that turns content-style
/// tokens into a mel in a reference's timbre (Vevo-Timbre), voiced by Vocos.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The paper leaves the sizes to the release (Amphion <c>models/vc/vevo</c>, the <c>amphion/Vevo</c> checkpoints),
/// whose code fills the gaps. <see cref="Synthesize"/> is Vevo-TTS: the voice's recording gives the style (with its
/// transcript's phones as the prompt) and, unless <see cref="TimbreReference"/> is set, the timbre.
/// <see cref="ConvertVoice"/>, <see cref="ConvertStyle"/> and <see cref="ConvertTimbre"/> are the voice-conversion
/// tasks.</para>
/// <para><b>For Beginners:</b> Vevo separates what is said, how it is said (accent, emotion) and who says it. Give it text and
/// a recording to read the text in that voice, or give it a recording to change its voice, its speaking style, or only its
/// timbre.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new Vevo&lt;float&gt;(architecture, new VevoOptions());
/// model.LoadReferenceWeights(new VevoCheckpoints { Directory = "Vevo", Hubert = "hubert-large-ll60k/pytorch_model.bin" });
/// model.Voice = model.CreateVoice(reference, referenceRate, "What the reference says.", "en");
/// var speech = model.Synthesize("Text to speak in that voice.");          // 24 kHz
/// var converted = model.ConvertVoice(source, sourceRate, reference, referenceRate);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement", "https://arxiv.org/abs/2502.07243",
    Year = 2025, Authors = "Xueyao Zhang, Xiaohui Zhang, Kainan Peng, Zhenyu Tang, Vimal Manohar, Yingru Liu, Jeff Hwang, Dangna Li, Yuhao Wang, Julian Chan, Yuan Huang, Zhizheng Wu, Mingbo Ma")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Source = "Amphion's Vevo trainers (egs/vc/AutoregressiveTransformer/ar_synthesis.json, FlowMatchingTransformer/fm_contentstyle.json, "
             + "models/base/base_trainer.py): AdamW with PyTorch's defaults at 2e-4 for the autoregressive transformers (gradient norm "
             + "clipped at 1) and 7.5e-5 for the flow (clipped at 0.2), transformers' inverse-square-root schedule after 32,000 warm-up steps.")]
public partial class Vevo<T> : TtsModelBase<T>
{
    private readonly VevoOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private HubertNetwork<T>? _hubert;
    private MaskGctSemanticCodec<T>? _contentStyle;
    private VevoContentCodec<T>? _content;
    private VevoAutoregressive<T>? _textAr;
    private VevoAutoregressive<T>? _voiceAr;
    private VevoFlowMatching<T>? _flow;
    private VocosGenerator<T>? _vocoder;
    private readonly List<LayerBase<T>> _hubertLayers = new();
    private readonly List<LayerBase<T>> _contentStyleLayers = new();
    private readonly List<LayerBase<T>> _contentLayers = new();
    private readonly List<LayerBase<T>> _textArLayers = new();
    private readonly List<LayerBase<T>> _voiceArLayers = new();
    private readonly List<LayerBase<T>> _flowLayers = new();

    private AmphionPhonemizer<T>? _phonemizer;
    private DifferentiableMel<T>? _mel;

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed Vevo for inference.</summary>
    public Vevo(NeuralNetworkArchitecture<T> architecture, string modelPath, VevoOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VevoOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Vevo.</summary>
    public Vevo(NeuralNetworkArchitecture<T> architecture, VevoOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VevoOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.HiddenDim % o.FlowHeads != 0) throw new ArgumentException("HiddenDim must divide by FlowHeads.");
        if (o.HubertLayer < 1) throw new ArgumentException("HubertLayer counts HuBERT's layers from 1.");
        if (o.HubertConvStrides.Aggregate(1, (a, b) => a * b) * o.SampleRate != o.HopSize * o.ContentSampleRate)
            throw new ArgumentException("HuBERT's frame rate must equal the mel's: the product of HubertConvStrides at ContentSampleRate must match HopSize at SampleRate.");
        if (o.MinNewTokens < 0 || o.MaxLength < 1) throw new ArgumentException("MinNewTokens is non-negative and MaxLength positive.");
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>Synthesis imitates a recording of the speaker (<see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool HasPaperLayers => _flow is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal HubertNetwork<T> Hubert => _hubert ?? throw NotNative();
    internal MaskGctSemanticCodec<T> ContentStyleTokenizer => _contentStyle ?? throw NotNative();
    internal VevoContentCodec<T> ContentTokenizer => _content ?? throw new InvalidOperationException("Voice conversion is off (VevoOptions.VoiceConversion).");
    internal VevoAutoregressive<T> TextToContentStyle => _textAr ?? throw NotNative();
    internal VevoAutoregressive<T> ContentToContentStyle => _voiceAr ?? throw new InvalidOperationException("Voice conversion is off (VevoOptions.VoiceConversion).");
    internal VevoFlowMatching<T> Flow => _flow ?? throw NotNative();
    internal VocosGenerator<T> Vocoder => _vocoder ?? throw NotNative();

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.PhoneVocabulary) : base.GetInputDomain(inputShape);

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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.TrainingSeed);
        _hubert = new HubertNetwork<T>(Engine, _hubertLayers, HubertConfiguration(o));
        _hubert.InitializeLikeReference(random);
        _contentStyle = new MaskGctSemanticCodec<T>(Engine, _contentStyleLayers, o.HubertHidden, o.ContentStyleVocosDim, o.ContentStyleVocosIntermediate,
            o.ContentStyleVocosLayers, o.ContentStyleCodes, o.ContentStyleCodeDim, random);
        _textAr = new VevoAutoregressive<T>(Engine, _textArLayers, o.PhoneVocabulary, o.ContentStyleCodes, o.TextArHidden, o.TextArIntermediate,
            o.TextArLayers, o.TextArHeads, random);
        if (o.VoiceConversion)
        {
            _content = new VevoContentCodec<T>(Engine, _contentLayers, new VevoContentCodecConfiguration(o.HubertHidden, o.ContentCodecChannels,
                o.ContentCodecChannels, o.ContentCodes, new[] { 1, 1 }, new[] { 1, 1 }), random);
            _voiceAr = new VevoAutoregressive<T>(Engine, _voiceArLayers, o.PhoneVocabulary + o.ContentCodes, o.ContentStyleCodes, o.VoiceArHidden,
                o.VoiceArIntermediate, o.VoiceArLayers, o.VoiceArHeads, random);
        }
        _flow = new VevoFlowMatching<T>(Engine, _flowLayers, new VevoFlowConfiguration(o.MelChannels, o.HiddenDim, o.FlowLayers, o.FlowHeads,
            o.ContentStyleCodes, o.FlowPromptDropProbability, o.FlowSigma, CosineSchedule: true), random);
        _vocoder = new VocosGenerator<T>(Engine, random, o.MelChannels, o.VocoderDim, o.VocoderIntermediate, o.VocoderLayers, o.FftSize, o.HopSize,
            samePadding: true);
        // The two autoregressive transformers and the flow train; HuBERT, both tokenizers and Vocos are pretrained and frozen.
        AddUnsplitLayers(_textArLayers.Concat(_voiceArLayers).Concat(_flowLayers).Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_hubertLayers);
        ComponentLayers.AddRange(_contentStyleLayers);
        ComponentLayers.AddRange(_contentLayers);
        ComponentLayers.AddRange(_vocoder.Layers);
    }

    internal static HubertConfiguration HubertConfiguration(VevoOptions o) => new()
    {
        ConvDims = o.HubertConvDims, ConvKernels = o.HubertConvKernels, ConvStrides = o.HubertConvStrides, LayerNormFeatureEncoder = true,
        HiddenSize = o.HubertHidden, Layers = o.HubertLayer, Heads = o.HubertHeads, IntermediateSize = o.HubertIntermediate,
        PositionalConvKernel = o.HubertPositionKernel, PositionalConvGroups = o.HubertPositionGroups, LayerNormFirst = true,
    };

    // ---------------------------------------------------------------- text

    /// <summary>Amphion's <c>g2p_</c>: the phone string and the phone ids of <paramref name="text"/> in
    /// <paramref name="language"/>.</summary>
    public (string Phonemes, int[] Ids) Phonemize(string text, string language) =>
        (_phonemizer ??= new AmphionPhonemizer<T>(Engine, () => _options.PolyphoneModelPath, () => _options.OpenJTalkDictionaryPath, nameof(VevoOptions)))
        .Phonemize(text, language);

    /// <inheritdoc />
    /// <remarks>The text's phone ids in <see cref="VevoOptions.Language"/>.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Phonemize(text, _options.Language).Ids;
        if (ids.Length == 0) throw new ArgumentException("The text has no phones.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    private int[] PhoneIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.PhoneVocabulary)
                throw new ArgumentOutOfRangeException(name, $"Phone ids lie in [0, {_options.PhoneVocabulary}).");
        }
        return ids;
    }

    // ---------------------------------------------------------------- audio features

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    /// <summary>A recording at the model's rate as <c>load_wav</c> makes it (librosa's soxr resampling).</summary>
    internal Tensor<T> AtModelRate(Tensor<T> recording, int sampleRate) =>
        sampleRate == _options.SampleRate ? Wave(Doubles(recording)) : Wave(SoxrResampler.LibrosaResample(Doubles(recording), sampleRate, _options.SampleRate));

    /// <summary>The 16 kHz copy HuBERT reads (torchaudio's resampling from the model's rate).</summary>
    internal Tensor<T> AtContentRate(Tensor<T> audio) =>
        AudioHelper<T>.ResampleBandLimited(Engine, Engine.Reshape(audio, new[] { audio.Length }), _options.SampleRate, _options.ContentSampleRate);

    /// <summary>The normalized mel <c>[frames, mel]</c> of audio at the model's rate (Amphion's <c>MelSpectrogram</c>,
    /// minus the mean over the standard deviation).</summary>
    internal Tensor<T> Mel(Tensor<T> audio)
    {
        var o = _options;
        _mel ??= new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.FftSize, o.MelChannels, 0, o.MelMaxFrequency, singlePrecisionBasis: true);
        var mel = _mel.Forward(Engine.Reshape(audio, new[] { audio.Length }));
        return Engine.TensorMultiplyScalar(Engine.TensorAddScalar(mel, Ops.FromDouble(-o.MelMean)), Ops.FromDouble(1.0 / Math.Sqrt(o.MelVariance)));
    }

    /// <summary>HuBERT's <see cref="VevoOptions.HubertLayer"/>-th layer output <c>[frames, hidden]</c> of 16 kHz audio.</summary>
    internal Tensor<T> HubertFeatures(Tensor<T> audio16k)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _hubertLayers) layer.SetTrainingMode(false);
        return Hubert.Forward(audio16k, _options.HubertLayer)[_options.HubertLayer];
    }

    /// <summary>The content-style tokens of 16 kHz audio (RepCodec over the normalized features).</summary>
    internal int[] ContentStyleCodes(Tensor<T> audio16k)
    {
        using var _ = new NoGradScope<T>();
        return ContentStyleTokenizer.Encode(HubertFeatures(audio16k));
    }

    /// <summary>The content tokens of 16 kHz audio, consecutive repeats removed.</summary>
    internal int[] ContentCodes(Tensor<T> audio16k)
    {
        using var _ = new NoGradScope<T>();
        return VevoContentCodec<T>.RemoveRepeats(ContentTokenizer.Codes(HubertFeatures(audio16k)));
    }

    // ---------------------------------------------------------------- references

    internal sealed class Reference
    {
        public Tensor<T> Audio = null!;       // at the model's rate
        public Tensor<T> Audio16k = null!;
        public int[]? ContentStyle;
        public Tensor<T>? Mel;
    }

    private readonly ConditionalWeakTable<Tensor<T>, Reference> _references = new();

    internal Reference ReferenceOf(Tensor<T> audio)
    {
        if (_references.TryGetValue(audio, out var r)) return r;
        r = new Reference { Audio = audio, Audio16k = AtContentRate(audio) };
        _references.Add(audio, r);
        return r;
    }

    private int[] ContentStyleOf(Reference r) => r.ContentStyle ??= ContentStyleCodes(r.Audio16k);

    private Tensor<T> MelOf(Reference r)
    {
        if (r.Mel is null)
            using (new NoGradScope<T>()) r.Mel = Mel(r.Audio);
        return r.Mel;
    }

    /// <summary>
    /// A voice from a recording of the speaker at <paramref name="sampleRate"/> and what it says: the recording gives the
    /// style (and, unless <see cref="TimbreReference"/> is set, the timbre); its transcript's phones in
    /// <paramref name="language"/> prompt the text-to-content-style transformer.
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string transcript, string language)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (transcript is null) throw new ArgumentNullException(nameof(transcript));
        if (!HasPaperLayers) throw NotNative();
        return new TtsVoice<T> { ReferenceAudio = AtModelRate(recording, sampleRate), ReferenceTokens = IndexTtsOps.Ids<T>(Phonemize(transcript, language).Ids) };
    }

    /// <summary>A recording whose timbre synthesis takes (at <see cref="TtsModelOptions.SampleRate"/>); null for the voice's
    /// own.</summary>
    public Tensor<T>? TimbreReference { get; set; }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return PredictCore(PreprocessText(text));
    }

    private VevoDecoding Decoding()
    {
        var o = _options;
        return new VevoDecoding(o.Sample, o.Temperature, o.TopK, o.TopP, o.RepetitionPenalty, o.MinNewTokens, o.MaxLength,
            o.ConstrainToContentStyleTokens);
    }

    /// <summary>The flow's mel for content-style tokens, prompted by the timbre reference's tokens and mel, through Vocos.</summary>
    internal Tensor<T> Render(IReadOnlyList<int> codes, Reference timbre, Random random)
    {
        var o = _options;
        var condition = Flow.Embed(ContentStyleOf(timbre).Concat(codes).ToArray());
        var prompt = MelOf(timbre);
        int target = condition.Shape[0] - prompt.Shape[0];
        if (target < 1) throw new InvalidOperationException("Nothing to generate: the timbre reference's mel is as long as the whole condition.");
        var noise = new Tensor<T>(new[] { target, o.MelChannels });
        for (int i = 0; i < noise.Length; i++) noise[i] = Ops.FromDouble(IndexTtsOps.Gaussian(random));
        var mel = Flow.Solve(condition, prompt, noise, o.FlowSteps, o.Guidance, o.GuidanceRescale);
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, o.MelChannels, target }));
        return Wave(Doubles(wave));
    }

    private T2 Inference<T2>(Func<Random, T2> run)
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return run(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's phone ids; the result is Vevo-TTS's speech at 24 kHz
    /// (<c>inference_ar_and_fm</c>: the reference transcript's phones then the text's, prompted by the reference's
    /// content-style tokens, to content-style tokens; the flow in the timbre reference's voice; Vocos).</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var phones = PhoneIds(input, nameof(input));
        if (phones.Length == 0) throw new ArgumentException("There are no phones.", nameof(input));
        var voice = RequireVoice();
        var styleAudio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} imitates a recording; set ReferenceAudio on the voice.");
        var referencePhones = voice.ReferenceTokens is { } tokens ? PhoneIds(tokens, nameof(voice)) : Array.Empty<int>();
        return Inference(random =>
        {
            var style = ReferenceOf(styleAudio);
            var codes = TextToContentStyle.Generate(referencePhones.Concat(phones).ToArray(), ContentStyleOf(style), Decoding(), random);
            return Render(codes, TimbreReference is { } timbre ? ReferenceOf(timbre) : style, random);
        });
    }

    /// <summary>The source's content tokens behind a style reference's, shifted past the phones (Vevo-Voice's input).</summary>
    private int[] VoiceInput(Reference style, Tensor<T> source)
    {
        var joined = Engine.TensorConcatenate(new[] { style.Audio16k, AtContentRate(source) }, 0);
        return ContentCodes(joined).Select(c => c + _options.PhoneVocabulary).ToArray();
    }

    /// <summary>
    /// Vevo-Voice: <paramref name="source"/> spoken in <paramref name="reference"/>'s voice — its timbre and its style
    /// (accent, emotion) — both at their own rates; the result is at 24 kHz.
    /// </summary>
    public Tensor<T> ConvertVoice(Tensor<T> source, int sourceRate, Tensor<T> reference, int referenceRate)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var sourceAudio = AtModelRate(source ?? throw new ArgumentNullException(nameof(source)), sourceRate);
        var referenceAudio = AtModelRate(reference ?? throw new ArgumentNullException(nameof(reference)), referenceRate);
        return Inference(random =>
        {
            var r = ReferenceOf(referenceAudio);
            var codes = ContentToContentStyle.Generate(VoiceInput(r, sourceAudio), ContentStyleOf(r), Decoding(), random);
            return Render(codes, r, random);
        });
    }

    /// <summary>Vevo-Style: <paramref name="source"/> in <paramref name="styleReference"/>'s style (accent, emotion),
    /// keeping its own timbre; the result is at 24 kHz.</summary>
    public Tensor<T> ConvertStyle(Tensor<T> source, int sourceRate, Tensor<T> styleReference, int styleRate)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var sourceAudio = AtModelRate(source ?? throw new ArgumentNullException(nameof(source)), sourceRate);
        var styleAudio = AtModelRate(styleReference ?? throw new ArgumentNullException(nameof(styleReference)), styleRate);
        return Inference(random =>
        {
            var style = ReferenceOf(styleAudio);
            var codes = ContentToContentStyle.Generate(VoiceInput(style, sourceAudio), ContentStyleOf(style), Decoding(), random);
            return Render(codes, ReferenceOf(sourceAudio), random);
        });
    }

    /// <summary>Vevo-Timbre: <paramref name="source"/> in <paramref name="timbreReference"/>'s timbre, keeping its content
    /// and style (the flow alone, <c>inference_fm</c>); the result is at 24 kHz.</summary>
    public Tensor<T> ConvertTimbre(Tensor<T> source, int sourceRate, Tensor<T> timbreReference, int timbreRate)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var sourceAudio = AtModelRate(source ?? throw new ArgumentNullException(nameof(source)), sourceRate);
        var timbreAudio = AtModelRate(timbreReference ?? throw new ArgumentNullException(nameof(timbreReference)), timbreRate);
        return Inference(random => Render(ContentStyleCodes(AtContentRate(sourceAudio)), ReferenceOf(timbreAudio), random));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Vevo-Native" : "Vevo-ONNX",
            Description = "Vevo: controllable zero-shot voice imitation with self-supervised disentanglement (Zhang et al., ICLR 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "Vevo";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Vevo<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
