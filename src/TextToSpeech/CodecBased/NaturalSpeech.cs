using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// NaturalSpeech (Tan et al. 2022): end-to-end text to waveform as a VAE whose posterior over the linear spectrogram is
/// simplified and whose prior from the phonemes is enhanced until they match: a pre-trained phoneme encoder, a
/// differentiable durator, a bidirectional prior/posterior flow trained with backward and forward KL losses, a VAE memory
/// bank, and an end-to-end loss on waveforms generated from the prior.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Inference (Algorithm 1): phonemes → hidden states → durations → the frame-level prior p(z′|y) by learnable
/// upsampling → z′ sampled → z = f(z′) → attention over the memory bank → the HiFi-GAN decoder.</para>
/// <para>Training follows App. F's stages (<see cref="NaturalSpeechOptions.CurrentStage"/>): mixed-phoneme pre-training,
/// warm-up with monotonic-alignment durations, the joint stage with the memory bank (initialized by K-means of posterior
/// latents) and both KL directions, and tuning of the durator with the end-to-end loss alone. Details the paper leaves out
/// come from heatz123/naturalspeech (<see cref="NaturalSpeechOptions.Heatz123"/>).</para>
/// <para><b>For Beginners:</b> NaturalSpeech reads English text aloud in the voice it was trained on, aiming for quality
/// listeners cannot tell from recordings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new NaturalSpeech&lt;float&gt;(architecture, new NaturalSpeechOptions());
/// // train: Warmup, then model.InitializeMemoryBank(recordings), then Joint, then Tuning
/// var audio = model.Synthesize("The quick brown fox.");   // 22.05 kHz; needs eSpeak NG installed
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality",
    "https://arxiv.org/abs/2205.04421",
    Year = 2022,
    Authors = "Xu Tan, Jiawei Chen, Haohe Liu, Jian Cong, Chen Zhang, Yanqing Liu, Xi Wang, Yichong Leng, Yuanhao Yi, Lei He, Frank Soong, Tao Qin, Sheng Zhao, Tie-Yan Liu"
)]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.8, Beta2 = 0.99, LearningRate = 2e-4, DecayRate = 0.999875,
                Source = "Tan et al. 2022, Training Details: AdamW with beta1 0.8, beta2 0.99, initial learning rate 2e-4, decayed by 0.999875 each epoch, for the generator and the discriminators.")]
public partial class NaturalSpeech<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    /// <summary>Calls the fully specified overload with the omitted parameters' defaults.</summary>
    public NaturalSpeech(NeuralNetworkArchitecture<T> architecture)
        : this(architecture, (NaturalSpeechOptions?)null)
    {
    }


    private readonly NaturalSpeechOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private NaturalSpeechCore<T>? _core;
    private HiFiGanDiscriminators<T>? _discriminators;
    private NaturalSpeechSupPhonemes? _supPhonemes;
    private EspeakPhonemizer? _phonemizer;
    private DifferentiableMel<T>? _spectrogram;

    private readonly List<LayerBase<T>> _phonemeLayers = new();
    private readonly List<LayerBase<T>> _duratorLayers = new();
    private readonly List<LayerBase<T>> _flowLayers = new();
    private readonly List<LayerBase<T>> _posteriorLayers = new();
    private readonly List<LayerBase<T>> _decoderLayers = new();
    private readonly List<LayerBase<T>> _memoryLayers = new();
    private readonly List<LayerBase<T>> _pretrainingLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed NaturalSpeech for inference.</summary>
    public NaturalSpeech(NeuralNetworkArchitecture<T> architecture, string modelPath, NaturalSpeechOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new NaturalSpeechOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable NaturalSpeech.</summary>
    public NaturalSpeech(NeuralNetworkArchitecture<T> architecture, NaturalSpeechOptions? options)
        : base(architecture)
    {
        _options = options ?? new NaturalSpeechOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        // VITS-style training clips nothing (clip_grad_value_(…, None)).
        MaxGradNorm = NumOps.Zero;
        if (o.HiddenDim % o.NumHeads != 0 || o.HiddenDim % 2 != 0) throw new ArgumentException("HiddenDim must be even and divide by NumHeads.");
        if (o.HiddenDim % o.MemoryHeads != 0) throw new ArgumentException("HiddenDim must divide by MemoryHeads.");
        if (o.UpsampleRates.Aggregate(1, (a, b) => a * b) != o.HopSize)
            throw new ArgumentException($"The decoder's upsampling rates must multiply to HopSize {o.HopSize}.");
        if (o.SegmentFrames < 1) throw new ArgumentException("SegmentFrames must be positive.");
        if (o.MaskRatio is <= 0 or >= 1) throw new ArgumentException("MaskRatio lies in (0, 1).");
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>The bidirectional prior/posterior's affine couplings.</remarks>
    public int NumFlowSteps => _options.Couplings;

    private bool HasPaperLayers => _core is not null;

    // The training and sampling seed (Options.Seed, 1234 when unset).
    private int SeedValue => _options.Seed ?? 1234;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal NaturalSpeechCore<T> Core => _core ?? throw NotNative();

    /// <summary>The symbol table: VITS's (keithito's) pad, punctuation, letters and IPA, 178 symbols.</summary>
    internal static readonly string Symbols = "_" + ";:,.!?¡¿—…\"«»“” " + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        + "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    private static readonly Dictionary<char, int> SymbolIds = BuildSymbolIds();

    private static Dictionary<char, int> BuildSymbolIds()
    {
        var ids = new Dictionary<char, int>();
        for (int i = 0; i < Symbols.Length; i++) ids[Symbols[i]] = i;
        return ids;
    }

    /// <summary>The space symbol's id (word boundaries for sup-phonemes).</summary>
    internal static int SpaceId => SymbolIds[' '];

    internal NaturalSpeechConfiguration Configuration() => new()
    {
        Vocab = _options.VocabSize, Hidden = _options.HiddenDim, Filter = _options.FilterChannels, Heads = _options.NumHeads,
        EncoderLayers = _options.NumEncoderLayers, EncoderKernel = _options.EncoderKernel, Dropout = _options.DropoutRate,
        Encoder = _options.PhonemeEncoder, MaxPositions = _options.MaxTextLength, DurationLayers = _options.DurationLayers,
        DurationFilter = _options.DurationFilter, DurationKernel = _options.DurationKernel, DurationDropout = _options.DurationDropout,
        Upsampling = _options.Upsampling, UpsamplingKernel = _options.UpsamplingKernel, UpsamplingConvOutput = _options.UpsamplingFilter,
        UpsamplingQ = _options.UpsamplingQ, UpsamplingP = _options.UpsamplingP, SpecChannels = _options.FftSize / 2 + 1, Couplings = _options.Couplings,
        CouplingLayers = _options.CouplingLayers, UpsampleRates = _options.UpsampleRates, UpsampleKernels = _options.UpsampleKernels,
        UpsampleInitialChannels = _options.UpsampleInitialChannels, ResblockKernels = _options.ResblockKernels,
        ResblockDilations = _options.ResblockDilations, MemorySize = _options.MemorySize, MemoryHeads = _options.MemoryHeads,
        SupPhonemes = SupPhonemes?.VocabSize ?? 0,
    };

    /// <summary>The sup-phoneme vocabulary of <see cref="NaturalSpeechOptions.SupPhonemeMerges"/>, or null.</summary>
    internal NaturalSpeechSupPhonemes? SupPhonemes => _supPhonemes ??= _options.SupPhonemeMerges is { } merges
        ? NaturalSpeechSupPhonemes.FromMerges(_options.VocabSize, merges) : null;

    /// <summary>
    /// Learns sup-phoneme merges (Mixed-Phoneme BERT's byte-pair encoding of adjacent phonemes inside words) from texts
    /// phonemized by <paramref name="phonemize"/> into symbol ids, until <paramref name="vocabSize"/> (the paper: 30,088)
    /// units; set the result as <see cref="NaturalSpeechOptions.SupPhonemeMerges"/> before building the model.
    /// </summary>
    public static IReadOnlyList<(int Left, int Right)> LearnSupPhonemes(IEnumerable<int[]> phonemeSequences, int phonemes, int vocabSize)
    {
        var words = phonemeSequences.SelectMany(SplitWords).Select(w => w.Ids);
        return NaturalSpeechSupPhonemes.Learn(words, phonemes, vocabSize).Merges.Select(m => (m.Left, m.Right)).ToList();
    }

    // A sequence's words (runs between spaces and punctuation-only symbols), each with its start.
    private static IEnumerable<(int Start, int[] Ids)> SplitWords(int[] sequence)
    {
        int start = 0;
        for (int i = 0; i <= sequence.Length; i++)
            if (i == sequence.Length || sequence[i] == SpaceId || sequence[i] == 0)
            {
                if (i > start) yield return (start, sequence.Skip(start).Take(i - start).ToArray());
                start = i + 1;
            }
    }

    /// <summary>The sup-phoneme id of each phoneme (spaces and the blank map to themselves), or null without sup-phonemes.</summary>
    internal int[]? SupPhonemeIds(int[] phonemes)
    {
        var sup = SupPhonemes;
        if (sup is null) return null;
        var ids = (int[])phonemes.Clone();
        foreach (var (start, word) in SplitWords(phonemes))
        {
            var assigned = sup.Assign(word);
            Array.Copy(assigned, 0, ids, start, assigned.Length);
        }
        return ids;
    }

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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? SeedValue);
        _core = new NaturalSpeechCore<T>(Engine, Configuration(), random, _phonemeLayers, _duratorLayers, _flowLayers, _posteriorLayers,
            _decoderLayers, _memoryLayers, _pretrainingLayers);
        _discriminators = new HiFiGanDiscriminators<T>(Engine, _options.DiscriminatorPeriods, 1, true, _options.DiscriminatorWidthDivisor, spectralFirstScale: false, vitsScale: true);
        AddUnsplitLayers(_phonemeLayers.Concat(_duratorLayers).Concat(_flowLayers).Concat(_decoderLayers).Concat(_memoryLayers)
            .Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_posteriorLayers);
        ComponentLayers.AddRange(_pretrainingLayers);
        ComponentLayers.AddRange(_discriminators.Layers);
    }

    // ---------------------------------------------------------------- text

    /// <summary>
    /// The text's symbol ids: keithito's english_cleaners2 (ASCII, lower case, abbreviations expanded, whitespace collapsed),
    /// eSpeak NG's phonemes through the phonemizer's espeak backend with punctuation and stress kept, the symbols looked up,
    /// and with <see cref="NaturalSpeechOptions.AddBlank"/> a blank between them.
    /// </summary>
    public int[] TextToPhonemes(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        _phonemizer ??= new EspeakPhonemizer(_options.EspeakLanguage, ";:,.!?¡¿—…\"«»“”", withStress: true);
        string cleaned = EnglishCleaners(text);
        string phonemes = Regex.Replace(_phonemizer.Phonemize(cleaned, "", " "), @"\s+", " ");
        var ids = phonemes.Where(SymbolIds.ContainsKey).Select(c => SymbolIds[c]).ToList();
        if (_options.AddBlank)
        {
            var blanked = new List<int> { 0 };
            foreach (int id in ids) { blanked.Add(id); blanked.Add(0); }
            ids = blanked;
        }
        return ids.ToArray();
    }

    private static readonly (Regex Pattern, string Replacement)[] Abbreviations = new[]
    {
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"), ("jr", "junior"), ("maj", "major"),
        ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"),
        ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    }.Select(a => (new Regex($@"\b{a.Item1}\.", RegexOptions.IgnoreCase), a.Item2)).ToArray();

    // english_cleaners2 before phonemization; unidecode's transliteration is approximated by dropping diacritics.
    private static string EnglishCleaners(string text)
    {
        var decomposed = text.Normalize(NormalizationForm.FormKD);
        var ascii = new StringBuilder();
        foreach (char c in decomposed)
            if (System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c) != System.Globalization.UnicodeCategory.NonSpacingMark && c < 128) ascii.Append(c);
        text = ascii.ToString().ToLowerInvariant();
        foreach (var (pattern, replacement) in Abbreviations) text = pattern.Replace(text, replacement);
        return Regex.Replace(text, @"\s+", " ");
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        return Wave(TextToPhonemes(text).Select(i => (double)i).ToArray());
    }

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(IReadOnlyList<double> samples) =>
        new(new[] { samples.Count }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    private int[] PhonemeIds(Tensor<T> input, string name)
    {
        var ids = new int[input.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(input.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(name, $"Phoneme ids lie in [0, {_options.VocabSize}).");
        }
        if (ids.Length == 0) throw new ArgumentException("There are no phonemes.", name);
        return ids;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the phoneme ids (<see cref="TextToPhonemes"/>); the result is the waveform at
    /// <see cref="TtsModelOptions.SampleRate"/> (Algorithm 1's inference).</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var phonemes = PhonemeIds(input, nameof(input));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var o = _options;
            var core = Core;
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
            var hidden = core.Encode(phonemes, SupPhonemeIds(phonemes), false, random);
            var logw = core.LogDurations(hidden, false, random);
            var w = Engine.TensorMultiplyScalar(Engine.TensorExp(logw), Ops.FromDouble(o.LengthScale));
            var (mean, logs, frames) = core.Upsampler.Forward(hidden, w, false, random);
            var noise = Gaussian(mean.Shape.ToArray(), random);
            var zPrime = Engine.TensorAdd(mean, Engine.TensorMultiplyScalar(Engine.TensorMultiply(noise, Engine.TensorExp(logs)), Ops.FromDouble(o.NoiseScale)));
            var z = core.Flow(zPrime, backward: false);
            return core.Decode(z);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    private static Tensor<T> Gaussian(int[] shape, Random random)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, Ops.FromDouble(AiDotNet.TextToSpeech.Latest.IndexTtsOps.Gaussian(random)));
        return t;
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "NaturalSpeech-Native" : "NaturalSpeech-ONNX",
            Description = "NaturalSpeech: end-to-end VAE text-to-waveform with a differentiable durator, bidirectional prior/posterior and memory VAE (Tan et al., 2022)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "NaturalSpeech";
        metadata.AdditionalInfo["Stage"] = _options.CurrentStage.ToString();
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(NaturalSpeech<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
