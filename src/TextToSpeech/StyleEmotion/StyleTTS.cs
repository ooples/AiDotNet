using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// StyleTTS: a style-based generative model that speaks text in the style of a reference recording (Li et al. 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The paper's eight modules (§II-A, Appendix D; reference yl4579/StyleTTS <c>models.py</c>): a text encoder (three
/// convolutions and a BiLSTM over phonemes); a style encoder (four residual blocks and average pooling over a mel
/// spectrogram, giving a 128-dimensional style); a decoder of seven residual blocks whose AdaIN layers read the style and
/// which also reads the aligned phoneme features, pitch and energy; a duration predictor and a prosody predictor (pitch
/// and energy) conditioned on the style through AdaLN and AdaIN; and, for training only, a discriminator, a text aligner
/// (the transferable monotonic aligner, pre-trained for ASR) and a pitch extractor (a JDC network). The output is a mel
/// spectrogram; a 24 kHz HiFi-GAN turns it into audio.
/// </para>
/// <para>
/// Training has two stages (§II-B; <see cref="CurrentStage"/>): reconstruction of the mel spectrogram from the aligned
/// phonemes, extracted pitch and energy and the style, adversarially and with the aligner's objectives; then the duration
/// and prosody predictors, on recordings stretched in time. Synthesis reads the style from a reference recording
/// (<see cref="TtsModelBase{T}.Voice"/>).
/// </para>
/// <para><b>For Beginners:</b> Give the model a short recording of how you want the speech to sound and the text to say;
/// it copies the speaking style — pitch movement, speed, emotion — onto the new sentence.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis",
    "https://arxiv.org/abs/2205.15439",
    Year = 2022,
    Authors = "Li, Han and Mesgarani"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.0, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 1e-4,
    Component = "generator", Provenance = RecipeProvenance.Stated,
    Source = "Section III-B: AdamW with beta1 = 0, beta2 = 0.99, weight decay 1e-4 and learning rate 1e-4 for both stages; "
             + "epsilon 1e-9 is the released code's (optimizers.py), whose one-cycle schedule with div and final-div factors "
             + "of 1 holds the rate constant.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.0, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 1e-4,
    Component = "discriminator", Provenance = RecipeProvenance.Stated,
    Source = "Section III-B (every module, the discriminator included, trains with the same AdamW).")]
public partial class StyleTTS<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly StyleTTSOptions _options;
    private readonly bool _useNativeMode;
    private readonly Random _trainingRandom;
    private bool _disposed;

    private StyleTtsTextEncoder<T>? _textEncoder;
    private StyleTtsStyleEncoder<T>? _styleEncoder;
    private StyleTtsDecoder<T>? _decoder;
    private StyleTtsProsodyPredictor<T>? _predictor;
    private StyleTtsTextAligner<T>? _aligner;
    private StyleTtsPitchExtractor<T>? _pitch;
    private StyleTtsDiscriminator<T>? _discriminator;
    private readonly List<LayerBase<T>> _textEncoderLayers = new();
    private readonly List<LayerBase<T>> _styleLayers = new();
    private readonly List<LayerBase<T>> _decoderLayers = new();
    private readonly List<LayerBase<T>> _predictorLayers = new();
    private readonly List<LayerBase<T>> _alignerLayers = new();
    private readonly List<LayerBase<T>> _pitchLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private StyleTtsFrontEnd<T>? _frontEnd;

    /// <summary>The number of entries in the released symbol table (178).</summary>
    public const int SymbolCount = StyleTtsFrontEnd<T>.SymbolCount;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed StyleTTS for inference.</summary>
    public StyleTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, StyleTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new StyleTTSOptions();
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable StyleTTS.</summary>
    public StyleTTS(NeuralNetworkArchitecture<T> architecture, StyleTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new StyleTTSOptions();
        _useNativeMode = true;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
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
        if (o.HiddenDim % 2 != 0) throw new ArgumentException("StyleTTS's BiLSTMs split the hidden width in half; use an even HiddenDim.");
        if (o.WindowLength > o.FftSize) throw new ArgumentException("The analysis window must fit the FFT.");
        // The reference clips no gradient.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int MelChannels => _options.MelChannels;

    /// <inheritdoc />
    public new int HopSize => _options.HopSize;

    /// <inheritdoc />
    public int FftSize => _options.FftSize;

    /// <summary>The mel spectrogram <c>[frames, mels]</c> of <paramref name="text"/> in the voice's style (pair it with a
    /// 24 kHz vocoder).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <summary>The modules a training step updates.</summary>
    public StyleTtsTrainingStage CurrentStage { get; set; } = StyleTtsTrainingStage.Reconstruction;

    /// <summary>Reconstruction updates taken (they decide when the aligner's objectives start).</summary>
    public int ReconstructionUpdates { get; private set; }

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice => TtsSupervision.SpeakerReference;

    /// <inheritdoc />
    /// <remarks>Its synthesized length follows its predicted durations.</remarks>
    public override bool TrainsOnSampleObjective => true;

    private bool HasPaperLayers => _textEncoder is not null;

    private StyleTtsTextEncoder<T> TextEncoderModule => _textEncoder ?? throw NotNative();
    private StyleTtsStyleEncoder<T> StyleEncoderModule => _styleEncoder ?? throw NotNative();
    private StyleTtsDecoder<T> DecoderModule => _decoder ?? throw NotNative();
    private StyleTtsProsodyPredictor<T> PredictorModule => _predictor ?? throw NotNative();
    private StyleTtsTextAligner<T> AlignerModule => _aligner ?? throw NotNative();
    private StyleTtsPitchExtractor<T> PitchModule => _pitch ?? throw NotNative();
    private StyleTtsDiscriminator<T> DiscriminatorModule => _discriminator ?? throw NotNative();
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    // The modules, for tests.
    internal StyleTtsTextEncoder<T> TextEncoderNetwork => TextEncoderModule;
    internal StyleTtsStyleEncoder<T> StyleEncoderNetwork => StyleEncoderModule;
    internal StyleTtsDecoder<T> DecoderNetwork => DecoderModule;
    internal StyleTtsProsodyPredictor<T> PredictorNetwork => PredictorModule;
    internal StyleTtsTextAligner<T> AlignerNetwork => AlignerModule;
    internal StyleTtsPitchExtractor<T> PitchExtractorNetwork => PitchModule;
    internal StyleTtsDiscriminator<T> DiscriminatorNetwork => DiscriminatorModule;

    // The modules' layers, for tests.
    internal IReadOnlyList<LayerBase<T>> TextEncoderLayers => _textEncoderLayers;
    internal IReadOnlyList<LayerBase<T>> StyleEncoderLayers => _styleLayers;
    internal IReadOnlyList<LayerBase<T>> DecoderLayers => _decoderLayers;
    internal IReadOnlyList<LayerBase<T>> PredictorLayers => _predictorLayers;
    internal IReadOnlyList<LayerBase<T>> AlignerLayers => _alignerLayers;
    internal IReadOnlyList<LayerBase<T>> PitchExtractorLayers => _pitchLayers;
    internal IReadOnlyList<LayerBase<T>> DiscriminatorLayers => _discriminatorLayers;

    /// <inheritdoc />
    /// <remarks>The phoneme symbol ids.</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        _textEncoder = new StyleTtsTextEncoder<T>(Engine, _textEncoderLayers, o.HiddenDim, o.TextKernel, o.TextLayers, o.VocabSize);
        _styleEncoder = new StyleTtsStyleEncoder<T>(Engine, _styleLayers, o.StyleEncoderDim, o.StyleDim, o.MaxConvDim);
        _decoder = new StyleTtsDecoder<T>(Engine, _decoderLayers, o.HiddenDim, o.StyleDim, o.ResidualDim, o.MelChannels);
        _predictor = new StyleTtsProsodyPredictor<T>(Engine, _predictorLayers, o.StyleDim, o.HiddenDim, o.TextLayers, o.DropoutRate);
        _aligner = new StyleTtsTextAligner<T>(Engine, _alignerLayers, o.MelChannels, o.AlignerHidden, o.VocabSize, o.AlignerTokenEmbedding, o.AlignerConvLayers);
        _pitch = new StyleTtsPitchExtractor<T>(Engine, _pitchLayers, o.PitchExtractorChannels, o.PitchExtractorLstm, o.MelChannels);
        _discriminator = new StyleTtsDiscriminator<T>(Engine, _discriminatorLayers, o.StyleEncoderDim, o.MaxConvDim);
        // The duration head is LinearNorm: Xavier-uniform weights (gain 1).
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        double bound = Math.Sqrt(6.0 / (o.HiddenDim + 1));
        TorchParameters.Linear(Engine, _predictor.DurationProjection, o.HiddenDim, 1,
            Enumerable.Range(0, o.HiddenDim).Select(_ => (2 * random.NextDouble() - 1) * bound).ToArray(), new double[1]);

        MaterializeModules();

        AddEncoderDecoderLayers(_textEncoderLayers.Cast<ILayer<T>>().ToList(), _decoderLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_styleLayers);
        ComponentLayers.AddRange(_predictorLayers);
        ComponentLayers.AddRange(_alignerLayers);
        ComponentLayers.AddRange(_pitchLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
    }

    /// <summary>
    /// Runs every module once on placeholder inputs, in inference mode and outside the tape, so the dense layers that
    /// size themselves on their first forward hold their weights from construction: the parameters a caller counts,
    /// saves or loads into are the ones the model trains.
    /// </summary>
    private void MaterializeModules()
    {
        var o = _options;
        var all = _textEncoderLayers.Concat(_styleLayers).Concat(_decoderLayers).Concat(_predictorLayers).Concat(_alignerLayers)
            .Concat(_pitchLayers).Concat(_discriminatorLayers).ToList();
        foreach (var layer in all) layer.SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = new Random(0);
            int frames = 80, tokens = 2;
            var mel = new Tensor<T>(new[] { o.MelChannels, frames });
            var ids = new[] { 0, 0 };
            var text = _textEncoder!.Forward(ToTensor(ids), false, random);
            var style = _styleEncoder!.Forward(mel);
            var encoded = _predictor!.EncodeText(text, style, false, random);
            _predictor.Durations(encoded, false, random);
            var aligned = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(encoded), new Tensor<T>(new[] { tokens, frames / 2 })),
                new[] { 1, o.HiddenDim + o.StyleDim, frames / 2 });
            var (f0, energy) = _predictor.Prosody(aligned, style, false, random);
            var asr = new Tensor<T>(new[] { 1, o.HiddenDim, frames / 2 });
            _decoder!.Forward(asr, f0, energy, style, false, random);
            _aligner!.Decode(mel, ids, false, random, 0.0);
            _aligner.CtcLogits(mel, false, random);
            _pitch!.Forward(mel, false, random);
            _discriminator!.Forward(Engine.Reshape(mel, new[] { 1, 1, o.MelChannels, frames }));
        }
        finally
        {
            foreach (var layer in all) layer.SetTrainingMode(true);
        }
    }

    // ---------------------------------------------------------------- front end

    /// <summary>The symbol ids of IPA phonemes as espeak writes them (words separated by spaces), framed by the pad
    /// symbol as the reference frames them; characters outside the table are skipped, as the reference skips them.</summary>
    public int[] EncodePhonemes(string phonemes)
    {
        return StyleTtsFrontEnd<T>.EncodePhonemes(phonemes);
    }

    /// <inheritdoc />
    /// <remarks>The text's espeak-style IPA phonemes with stress (<see cref="EnglishG2P"/>), as symbol ids framed by the
    /// pad symbol.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var builder = new System.Text.StringBuilder();
        foreach (var symbol in EnglishG2P.Default.Phonemize(text))
            builder.Append(symbol == EnglishG2P.WordSeparator ? " " : symbol);
        var ids = EncodePhonemes(builder.ToString());
        if (ids.Length <= 2) throw new ArgumentException("The text has no pronounceable content.", nameof(text));
        return ToTensor(ids.Take(_options.MaxTextLength).ToArray());
    }

    private Tensor<T> ToTensor(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private int[] ToIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"{GetType().Name}'s symbol ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <summary>
    /// The model's mel spectrogram <c>[mels, frames]</c> of a waveform at its sample rate: torchaudio's
    /// <c>MelSpectrogram</c> (centred STFT, periodic Hann window, power 2, HTK mel scale), then
    /// <c>(log(1e-5 + mel) − mean) / std</c>, as the reference preprocesses.
    /// </summary>
    public Tensor<T> ComputeMel(Tensor<T> audio) => FrontEnd.ComputeMel(audio);

    /// <summary>The energy of each frame of a normalized mel spectrogram <c>[mels, frames]</c>: the log of the L2 norm of
    /// the de-normalized mel magnitudes (the reference's <c>log_norm</c>).</summary>
    internal Tensor<T> LogNorm(Tensor<T> mel) => FrontEnd.LogNorm(mel);

    private StyleTtsFrontEnd<T> FrontEnd => _frontEnd ??= new StyleTtsFrontEnd<T>(Engine, _options);

    /// <summary>A voice for <see cref="TtsModelBase{T}.Voice"/>: a reference recording at the model's sample rate whose
    /// speaking style the synthesized speech takes.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording)
    {
        var mel = ComputeMel(recording ?? throw new ArgumentNullException(nameof(recording)));
        return new TtsVoice<T> { Reference = Engine.TensorTranspose(mel), ReferenceAudio = recording };
    }

    /// <summary>The style vector <c>[1, style]</c> of a reference mel spectrogram <c>[frames, mels]</c>.</summary>
    public Tensor<T> ComputeStyle(Tensor<T> referenceMel)
    {
        if (referenceMel is null) throw new ArgumentNullException(nameof(referenceMel));
        using var _ = new NoGradScope<T>();
        return StyleEncoderModule.Forward(MelsFirst(referenceMel));
    }

    // [frames, mels] (the library's spectrogram layout) as [mels, frames] (the reference's).
    private Tensor<T> MelsFirst(Tensor<T> mel)
    {
        if (mel.Rank != 2 || mel.Shape[1] != _options.MelChannels)
            throw new ArgumentException($"Expected a mel spectrogram [frames, {_options.MelChannels}].", nameof(mel));
        return Engine.TensorTranspose(mel);
    }

    // ---------------------------------------------------------------- synthesis

    /// <summary>The 0/1 alignment <c>[phonemes, frames]</c> of whole-frame durations.</summary>
    private Tensor<T> AlignmentOf(int[] durations)
    {
        int total = durations.Sum();
        var alignment = new Tensor<T>(new[] { durations.Length, total });
        int frame = 0;
        for (int i = 0; i < durations.Length; i++)
            for (int f = 0; f < durations[i]; f++) alignment[i, frame++] = NumOps.One;
        return alignment;
    }

    /// <summary>The mel spectrogram <c>[1, mels, 2F]</c> of phonemes in a style, with durations from the duration
    /// predictor (the reference's inference notebooks).</summary>
    internal Tensor<T> SynthesizeMel(int[] ids, Tensor<T> style)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var text = TextEncoderModule.Forward(ToTensor(ids), false, random);                                   // [1, C, L]
        var encoded = PredictorModule.EncodeText(text, style, false, random);                                // [L, C + S]
        var durations = PredictorModule.Durations(encoded, false, random);
        var frames = new int[ids.Length];
        for (int i = 0; i < frames.Length; i++) frames[i] = Math.Max(1, (int)Math.Round(NumOps.ToDouble(durations[i]), MidpointRounding.ToEven));
        var alignment = AlignmentOf(frames);                                                                 // [L, F]
        int width = _options.HiddenDim + _options.StyleDim, total = frames.Sum();
        var aligned = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(encoded), alignment), new[] { 1, width, total });
        var (f0, energy) = PredictorModule.Prosody(aligned, style, false, random);
        var asr = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(text, new[] { _options.HiddenDim, ids.Length }), alignment),
            new[] { 1, _options.HiddenDim, total });
        return DecoderModule.Forward(asr, f0, energy, style, false, random);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the phoneme symbol ids (<see cref="PreprocessText"/>); the output is the mel
    /// spectrogram <c>[frames, mels]</c> in the voice's style.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        var voice = RequireVoice();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var style = StyleEncoderModule.Forward(MelsFirst(voice.Reference!));
            var mel = SynthesizeMel(ToIds(input, nameof(input)), style);
            return Engine.TensorTranspose(Engine.Reshape(mel, new[] { _options.MelChannels, mel.Shape[2] }));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- pretrained weights

    internal static TorchReader Reader(string path, params string[] prefixes)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        return (name, shape) =>
        {
            AiDotNet.Tensors.LinearAlgebra.Tensor<float>? tensor = null;
            foreach (var prefix in prefixes.Append(""))
                foreach (var candidate in new[] { prefix + name, prefix + "module." + name })
                    if (tensor is null && file.TryGetValue(candidate, out var found)) tensor = found;
            if (tensor is null) throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var actual = tensor.Shape.ToArray();
            if (actual.Aggregate(1, (a, b) => a * b) != shape.Aggregate(1, (a, b) => a * b) || (actual.Length == shape.Length && !actual.SequenceEqual(shape)))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] in this model.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        };
    }

    /// <summary>
    /// Loads a StyleTTS checkpoint flattened to named tensors (safetensors, or a <c>.pt</c> state dict) whose names are
    /// the reference's module keys (<c>text_encoder</c>, <c>style_encoder</c>, <c>decoder</c>, <c>predictor</c>,
    /// <c>text_aligner</c>, <c>pitch_extractor</c> and optionally <c>discriminator</c>) and their parameters, with or
    /// without the checkpoint's <c>net.</c> prefix.
    /// </summary>
    public void LoadReferenceWeights(string path, bool includeDiscriminator = false) =>
        LoadReferenceWeights(Reader(path, "net."), includeDiscriminator);

    internal void LoadReferenceWeights(TorchReader read, bool includeDiscriminator = false)
    {
        TextEncoderModule.Load(read, "text_encoder");
        StyleEncoderModule.Load(Engine, read, "style_encoder");
        DecoderModule.Load(read, "decoder");
        PredictorModule.Load(read, "predictor");
        AlignerModule.Load(read, "text_aligner");
        PitchModule.Load(read, "pitch_extractor");
        if (includeDiscriminator) DiscriminatorModule.Load(read, "discriminator");
    }

    /// <summary>Loads the released text aligner (<c>Utils/ASR/epoch_00080.pth</c>'s <c>model</c> entry, saved as
    /// safetensors or <c>.pt</c>).</summary>
    public void LoadAlignerWeights(string path) => AlignerModule.Load(StripPrefix(Reader(path, "model."), "text_aligner."), "text_aligner");

    /// <summary>Loads the released pitch extractor (<c>Utils/JDC/bst.t7</c>'s <c>net</c> entry, saved as safetensors or
    /// <c>.pt</c>).</summary>
    public void LoadPitchExtractorWeights(string path) => PitchModule.Load(StripPrefix(Reader(path, "net."), "pitch_extractor."), "pitch_extractor");

    internal static TorchReader StripPrefix(TorchReader read, string prefix) =>
        (name, shape) => read(name.StartsWith(prefix, StringComparison.Ordinal) ? name.Substring(prefix.Length) : name, shape);

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "StyleTTS-Native" : "StyleTTS-ONNX",
            Description = "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis (Li et al., 2022)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.TextLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "StyleTTS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(StyleTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
