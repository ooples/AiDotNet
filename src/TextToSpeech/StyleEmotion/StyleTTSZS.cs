using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Speaker;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// StyleTTS-ZS: efficient high-quality zero-shot text-to-speech with distilled time-varying style diffusion (Li et al.
/// 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The paper's four modules (§III): an acoustic synthesizer — a joint prompt-text encoder (conformer blocks over the
/// phonemes and the prompt's mel-spectrogram, giving prompt-aligned text embeddings and a pooled global style) and
/// StyleTTS 2's decoder, which reads them with the phoneme durations, pitch and energy; a vector-quantized prosody
/// autoencoder — a prosody encoder compressing durations (as upsampled position embeddings), pitch and energy into a
/// fixed-length time-varying style (50 rows, 9 quantizer codebooks), and duration and pitch-energy decoders conditioned on
/// PL-BERT's phoneme embeddings; a style diffusion denoiser sampling that style from the text and the prompt (v-objective,
/// angular schedule, DDIM, classifier-free guidance) and a one-step student distilled from it; and multimodal
/// discriminators for the waveform (on WavLM features) and the prosody, besides StyleTTS 2's multi-period and
/// multi-resolution ones.
/// </para>
/// <para>
/// No code was released; where the paper is silent the model follows StyleTTS 2's code and the components the paper cites
/// (docs/text-to-speech/paper-fidelity-decisions.md lists each choice).
/// </para>
/// <para><b>For Beginners:</b> Give StyleTTS-ZS a few seconds of anyone's speech and some text; it speaks the text in
/// that voice, choosing a natural rhythm and intonation in one fast step.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "StyleTTS-ZS: Efficient High-Quality Zero-Shot Text-to-Speech Synthesis with Distilled Time-Varying Style Diffusion",
    "https://arxiv.org/abs/2409.10058",
    Year = 2024,
    Authors = "Li, Jiang, Han and Mesgarani"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.0, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 1e-4,
    Component = "generator", Provenance = RecipeProvenance.Stated,
    Source = "Section IV-A: AdamW with beta1 = 0, beta2 = 0.99, weight decay 1e-4 and learning rate 1e-4; epsilon 1e-9 is "
             + "StyleTTS 2's code's, which the paper otherwise follows.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.0, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 1e-4,
    Component = "discriminator", Provenance = RecipeProvenance.Stated,
    Source = "Section IV-A (every module trains with the same AdamW).")]
public partial class StyleTTSZS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    /// <summary>Calls the fully specified overload with the omitted parameters' defaults.</summary>
    public StyleTTSZS(NeuralNetworkArchitecture<T> architecture)
        : this(architecture, (StyleTTSZSOptions?)null)
    {
    }


    private readonly StyleTTSZSOptions _options;
    private readonly bool _useNativeMode;
    private readonly Random _trainingRandom;
    private bool _disposed;
    private StyleTtsFrontEnd<T>? _frontEnd;

    private StyleTtsZsPromptTextEncoder<T>? _promptText;
    private AlbertNetwork<T>? _bert;
    private DenseLayer<T>? _bertEncoder;
    private StyleTtsZsProsodyEncoder<T>? _prosodyEncoder;
    private FactorizedVectorQuantizerLayer<T>? _quantizer;
    private StyleTtsZsProsodyDecoder<T>? _durationDecoder;
    private StyleTtsZsProsodyDecoder<T>? _prosodyDecoder;
    private StyleTtsZsDenoiser<T>? _teacher;
    private StyleTtsZsDenoiser<T>? _student;
    private StyleTts2Decoder<T>? _decoder;
    private StyleTtsTextAligner<T>? _aligner;
    private StyleTtsPitchExtractor<T>? _pitch;
    private CosyVoiceHiFTDiscriminators<T>? _waveDiscriminators;
    private StyleTtsZsMultimodalDiscriminator<T>? _waveformDiscriminator;
    private StyleTtsZsMultimodalDiscriminator<T>? _durationDiscriminator;
    private StyleTtsZsMultimodalDiscriminator<T>? _prosodyDiscriminator;
    private WavLmNetwork<T>? _slm;
    private WeSpeakerResNet<T>? _speaker;

    private readonly Dictionary<string, List<LayerBase<T>>> _modules = new();

    private List<LayerBase<T>> Module(string key)
    {
        if (!_modules.TryGetValue(key, out var list)) _modules[key] = list = new List<LayerBase<T>>();
        return list;
    }

    /// <summary>Creates an ONNX-backed StyleTTS-ZS for inference.</summary>
    public StyleTTSZS(NeuralNetworkArchitecture<T> architecture, string modelPath, StyleTTSZSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new StyleTTSZSOptions();
        Options = _options;
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(_options.ModelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable StyleTTS-ZS.</summary>
    public StyleTTSZS(NeuralNetworkArchitecture<T> architecture, StyleTTSZSOptions? options)
        : base(architecture)
    {
        _options = options ?? new StyleTTSZSOptions();
        Options = _options;
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
        base.HiddenDim = o.ModelDim;
        if (o.ModelDim <= 0 || o.Heads <= 0 || o.ModelDim % o.Heads != 0)
            throw new ArgumentException("The positive model width must divide by the positive head count.");
        if (o.StyleDim != o.ModelDim)
            throw new ArgumentException("The global style is the prompt-text encoder's pooled output: StyleDim must equal ModelDim.");
        if (o.WindowLength > o.FftSize) throw new ArgumentException("The analysis window must fit the FFT.");
        if (o.UpsampleRates.Length == 0 || o.UpsampleRates.Length != o.UpsampleKernels.Length)
            throw new ArgumentException("Give one upsampling kernel per upsampling rate.");
        for (int i = 0; i < o.UpsampleRates.Length; i++)
            if (o.UpsampleRates[i] <= 0 || o.UpsampleKernels[i] != 2L * o.UpsampleRates[i])
                throw new ArgumentException("Each positive upsampling rate needs a kernel twice its size.", nameof(o.UpsampleKernels));
        int hop = o.UpsampleRates.Aggregate(1, (a, b) => a * b) * (o.DecoderType == StyleTts2DecoderType.IstftNet ? o.IstftHopSize : 1);
        if (hop != o.HopSize) throw new ArgumentException($"The generator's hop ({hop}) must equal the mel hop ({o.HopSize}).");
        if (o.Bert.VocabSize < o.VocabSize) throw new ArgumentException("PL-BERT must embed every phoneme symbol.");
        if (o.DistillationGuidanceMax < o.DistillationGuidanceMin) throw new ArgumentException("The distillation guidance range is empty.");
        // StyleTTS 2's code, which StyleTTS-ZS follows, clips no gradient.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int HiddenDim => _options.ModelDim;

    /// <inheritdoc />
    /// <remarks>StyleTTS-ZS has no flow; this is the teacher's DDIM steps (1 with the distilled student).</remarks>
    public int NumFlowSteps => _options.UseDistilledSampler ? 1 : _options.SamplingSteps;

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice => TtsSupervision.SpeakerReference;

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    private bool HasPaperLayers => _promptText is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal StyleTtsZsPromptTextEncoder<T> PromptTextNetwork => _promptText ?? throw NotNative();
    internal AlbertNetwork<T> BertNetwork => _bert ?? throw NotNative();
    internal DenseLayer<T> BertEncoderLayer => _bertEncoder ?? throw NotNative();
    internal StyleTtsZsProsodyEncoder<T> ProsodyEncoderNetwork => _prosodyEncoder ?? throw NotNative();
    internal FactorizedVectorQuantizerLayer<T> Quantizer => _quantizer ?? throw NotNative();
    internal StyleTtsZsProsodyDecoder<T> DurationDecoderNetwork => _durationDecoder ?? throw NotNative();
    internal StyleTtsZsProsodyDecoder<T> ProsodyDecoderNetwork => _prosodyDecoder ?? throw NotNative();
    internal StyleTtsZsDenoiser<T> TeacherNetwork => _teacher ?? throw NotNative();
    internal StyleTtsZsDenoiser<T> StudentNetwork => _student ?? throw NotNative();
    internal StyleTts2Decoder<T> DecoderNetwork => _decoder ?? throw NotNative();
    internal StyleTtsTextAligner<T> AlignerNetwork => _aligner ?? throw NotNative();
    internal StyleTtsPitchExtractor<T> PitchExtractorNetwork => _pitch ?? throw NotNative();
    internal CosyVoiceHiFTDiscriminators<T> WaveDiscriminators => _waveDiscriminators ?? throw NotNative();
    internal StyleTtsZsMultimodalDiscriminator<T> WaveformDiscriminator => _waveformDiscriminator ?? throw NotNative();
    internal StyleTtsZsMultimodalDiscriminator<T> DurationDiscriminator => _durationDiscriminator ?? throw NotNative();
    internal StyleTtsZsMultimodalDiscriminator<T> ProsodyDiscriminator => _prosodyDiscriminator ?? throw NotNative();
    internal WavLmNetwork<T> SlmNetwork => _slm ?? throw NotNative();
    internal WeSpeakerResNet<T> SpeakerNetwork => _speaker ?? throw NotNative();

    private StyleTtsFrontEnd<T> FrontEnd => _frontEnd ??= new StyleTtsFrontEnd<T>(Engine, _options);

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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        int d = o.ModelDim;
        double dropout = o.ConformerDropout;
        _promptText = new StyleTtsZsPromptTextEncoder<T>(Engine, Module("prompt_text_encoder"), o.VocabSize, o.MelChannels, d, o.Heads, o.FeedForwardDim,
            o.PromptTextKernels, dropout, o.Separators, random);
        _bert = new AlbertNetwork<T>(Engine, Module("bert"), o.Bert);
        _bertEncoder = StyleTtsOps.Own(Module("bert_encoder"), StyleTtsOps.Linear<T>(d));
        _prosodyEncoder = new StyleTtsZsProsodyEncoder<T>(Engine, Module("prosody_encoder"), o.MaxPhonemes, d, o.Heads, o.FeedForwardDim,
            o.ProsodyEncoderKernels, o.StyleLength, dropout, random);
        _quantizer = StyleTtsOps.Own(Module("prosody_encoder"), new FactorizedVectorQuantizerLayer<T>(d, o.Quantizers, o.CodebookSize, o.CodeDim, lossesOnRawVectors: true));
        _durationDecoder = new StyleTtsZsProsodyDecoder<T>(Engine, Module("duration_decoder"), d, o.Heads, o.FeedForwardDim, o.ProsodyDecoderKernels, 1,
            dropout, o.Separators, random);
        _prosodyDecoder = new StyleTtsZsProsodyDecoder<T>(Engine, Module("prosody_decoder"), d, o.Heads, o.FeedForwardDim, o.ProsodyDecoderKernels, 2,
            dropout, o.Separators, random);
        _teacher = new StyleTtsZsDenoiser<T>(Engine, Module("diffusion"), o.MelChannels, d, o.Heads, o.FeedForwardDim, o.DenoiserKernels, dropout,
            o.Separators, random);
        _student = new StyleTtsZsDenoiser<T>(Engine, Module("student"), o.MelChannels, d, o.Heads, o.FeedForwardDim, o.DenoiserKernels, dropout,
            o.Separators, random);
        _decoder = new StyleTts2Decoder<T>(Engine, Module("decoder"), o.DecoderType, d, o.StyleDim, o.DecoderBottleneck, o.ResidualDim, o.ResBlockKernels,
            o.UpsampleRates, o.UpsampleInitialChannels, o.ResBlockDilations, o.UpsampleKernels, o.IstftFftSize, o.IstftHopSize, o.SampleRate);
        _aligner = new StyleTtsTextAligner<T>(Engine, Module("text_aligner"), o.MelChannels, o.AlignerHidden, o.VocabSize, o.AlignerTokenEmbedding, o.AlignerConvLayers);
        _pitch = new StyleTtsPitchExtractor<T>(Engine, Module("pitch_extractor"), o.PitchExtractorChannels, o.PitchExtractorLstm, o.MelChannels);
        _waveDiscriminators = new CosyVoiceHiFTDiscriminators<T>(Engine);
        int periodLayers = _waveDiscriminators.Periods.Layers.Count;
        Module("mpd").AddRange(_waveDiscriminators.Layers.Take(periodLayers));
        Module("msd").AddRange(_waveDiscriminators.Layers.Skip(periodLayers));
        _waveformDiscriminator = new StyleTtsZsMultimodalDiscriminator<T>(Engine, Module("waveform_discriminator"), d, o.Heads, o.FeedForwardDim,
            o.DiscriminatorKernels, dropout, random);
        _durationDiscriminator = new StyleTtsZsMultimodalDiscriminator<T>(Engine, Module("duration_discriminator"), d, o.Heads, o.FeedForwardDim,
            o.DiscriminatorKernels, dropout, random);
        _prosodyDiscriminator = new StyleTtsZsMultimodalDiscriminator<T>(Engine, Module("prosody_discriminator"), d, o.Heads, o.FeedForwardDim,
            o.DiscriminatorKernels, dropout, random);
        _slm = new WavLmNetwork<T>(Engine, Module("wavlm"), o.SlmConfiguration);
        _speaker = new WeSpeakerResNet<T>(Engine, Module("speaker"), o.SpeakerBlocks, o.SpeakerChannels, embedding: o.SpeakerEmbedding);

        MaterializeModules();

        AddEncoderDecoderLayers(Module("prompt_text_encoder").Cast<ILayer<T>>().ToList(), Module("decoder").Cast<ILayer<T>>().ToList());
        foreach (var entry in _modules)
            if (entry.Key is not ("prompt_text_encoder" or "decoder")) ComponentLayers.AddRange(entry.Value);
    }

    /// <summary>
    /// Runs every module once on placeholder inputs, in inference mode and outside the tape, so the dense layers that
    /// size themselves on their first forward hold their weights from construction.
    /// </summary>
    private void MaterializeModules()
    {
        var o = _options;
        var all = _modules.Values.SelectMany(l => l).ToList();
        foreach (var layer in all) layer.SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0);
            int frames = 80, d = o.ModelDim;
            var mel = new Tensor<T>(new[] { o.MelChannels, frames });
            var ids = new[] { 0, 0 };
            var (text, style) = _promptText!.Forward(ids, mel, false, random);
            var e = _bertEncoder!.Forward(_bert!.Encode(ids, false, random));
            var phonemeOfFrame = PhonemeOfFrame(new[] { 1, 1 }, 2);
            var h = _quantizer!.Forward(PhonemesFirst(_prosodyEncoder!.Forward(phonemeOfFrame, new Tensor<T>(new[] { 4 }), new Tensor<T>(new[] { 4 }), false, random)));
            var styleRows = RowsFirst(h);
            _durationDecoder!.Forward(e, styleRows, false, random);
            _prosodyDecoder!.Forward(new Tensor<T>(new[] { 4, d }), styleRows, false, random);
            _teacher!.Forward(styleRows, 0.5, e, mel, false, false, random);
            _student!.Forward(styleRows, 1.0, e, mel, false, false, random);
            int half = 4;
            _decoder!.Forward(new Tensor<T>(new[] { 1, d, half }), new Tensor<T>(new[] { 1, 2 * half }), new Tensor<T>(new[] { 1, 2 * half }), style, false, random);
            _aligner!.Decode(mel, ids, false, random, 0.0);
            _aligner.CtcLogits(mel, false, random);
            _pitch!.Forward(mel, false, random);
            _waveDiscriminators!.Forward(new Tensor<T>(new[] { 2 * o.HopSize * 4 }));
            var states = _slm!.HiddenStates(new Tensor<T>(new[] { 800 }));
            _waveformDiscriminator!.Forward(new Tensor<T>(new[] { 2, o.SlmHidden * o.SlmLayers }), new Tensor<T>(new[] { 2, WaveformConditionWidth }), false, random);
            _durationDiscriminator!.Forward(new Tensor<T>(new[] { 2, 1 }), new Tensor<T>(new[] { 2, d }), false, random);
            _prosodyDiscriminator!.Forward(new Tensor<T>(new[] { 2, 2 }), new Tensor<T>(new[] { 2, d }), false, random);
            _speaker!.Forward(new Tensor<T>(new[] { 1600 }));
        }
        finally
        {
            foreach (var layer in all) layer.SetTrainingMode(true);
        }
    }

    /// <summary>The waveform discriminator's per-frame conditions: aligned text embeddings, the global style, pitch,
    /// energy and the frame's phoneme duration.</summary>
    private int WaveformConditionWidth => _options.ModelDim + _options.StyleDim + 3;

    // ---------------------------------------------------------------- front end and alignment helpers

    /// <summary>The symbol ids of IPA phonemes, framed by the pad symbol as StyleTTS 2's data frames them.</summary>
    public int[] EncodePhonemes(string phonemes) => StyleTtsFrontEnd<T>.EncodePhonemes(phonemes);

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var builder = new System.Text.StringBuilder();
        foreach (var symbol in EnglishG2P.Default.Phonemize(text))
            builder.Append(symbol == EnglishG2P.WordSeparator ? " " : symbol);
        var ids = EncodePhonemes(builder.ToString());
        if (ids.Length <= 2) throw new ArgumentException("The text has no pronounceable content.", nameof(text));
        return ToTensor(ids.Take(Math.Min(_options.MaxTextLength, Math.Min(_options.Bert.MaxPositions, _options.MaxPhonemes))).ToArray());
    }

    private Tensor<T> ToTensor(IReadOnlyList<int> ids)
    {
        return IdTensor(ids);
    }

    private int[] ToIds(Tensor<T> tensor, string name)
    {
        var ids = RoundedIds(tensor);
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0 || ids[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"{GetType().Name}'s symbol ids lie in [0, {_options.VocabSize}).");
        }
        int limit = Math.Min(_options.Bert.MaxPositions, _options.MaxPhonemes);
        if (ids.Length > limit) throw new ArgumentException($"At most {limit} symbols.", name);
        return ids;
    }

    /// <summary>The normalized log-mel spectrogram <c>[mels, frames]</c> of a waveform (StyleTTS's front end).</summary>
    public Tensor<T> ComputeMel(Tensor<T> audio) => FrontEnd.ComputeMel(audio);

    /// <summary>A voice: a few seconds of the target speaker at the model's sample rate (the evaluation uses 3 s).</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording)
    {
        var mel = ComputeMel(recording ?? throw new ArgumentNullException(nameof(recording)));
        return new TtsVoice<T> { Reference = Engine.TensorTranspose(mel), ReferenceAudio = recording };
    }

    private Tensor<T> MelsFirst(Tensor<T> mel)
    {
        if (mel.Rank != 2 || mel.Shape[1] != _options.MelChannels)
            throw new ArgumentException($"Expected a mel spectrogram [frames, {_options.MelChannels}].", nameof(mel));
        return Engine.TensorTranspose(mel);
    }

    private Tensor<T> PhonemesFirst(Tensor<T> rows) => Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    private Tensor<T> RowsFirst(Tensor<T> channelsFirst) =>
        Engine.TensorTranspose(Engine.Reshape(channelsFirst, new[] { channelsFirst.Shape[1], channelsFirst.Shape[2] }));

    /// <summary>The phoneme each frame belongs to, each phoneme's duration repeated <paramref name="factor"/> times per
    /// frame (2: the mel rate of aligner-frame durations).</summary>
    internal static int[] PhonemeOfFrame(int[] durations, int factor)
    {
        var frames = new List<int>();
        for (int i = 0; i < durations.Length; i++)
            for (int f = 0; f < factor * durations[i]; f++) frames.Add(i);
        return frames.ToArray();
    }

    /// <summary>Rows <c>[L, d]</c> upsampled to frames by <paramref name="phonemeOfFrame"/> (on the tape).</summary>
    private Tensor<T> Upsample(Tensor<T> rows, int[] phonemeOfFrame) =>
        Engine.TensorIndexSelect(rows, new Tensor<int>(new[] { phonemeOfFrame.Length }, new Vector<int>(phonemeOfFrame)), 0);

    /// <summary>The 0/1 alignment <c>[phonemes, frames]</c> of whole-frame durations.</summary>
    private Tensor<T> AlignmentOf(int[] durations)
    {
        return DurationAlignment(durations);
    }

    internal int[] FramesOf(Tensor<T> durations)
    {
        var frames = RoundedIds(durations);
        for (int i = 0; i < frames.Length; i++) frames[i] = Math.Max(1, frames[i]);
        return frames;
    }

    // ---------------------------------------------------------------- style sampling

    private static (double Alpha, double Sigma) Schedule(double tau) => (Math.Cos(Math.PI * tau / 2), Math.Sin(Math.PI * tau / 2));

    /// <summary>
    /// α_n = cos(πτ_n/2) and σ_n = sin(πτ_n/2) at τ_n = n / <paramref name="steps"/>, n = 0 … steps: each formed in double
    /// and rounded to T once, as the per-step scalars always were (in T, α at τ = 1 is cos of a rounded π/2, not 0).
    /// </summary>
    internal static (T[] Alpha, T[] Sigma) AngularSchedule(int steps)
    {
        var alpha = new double[steps + 1];
        var sigma = new double[steps + 1];
        for (int n = 0; n <= steps; n++) (alpha[n], sigma[n]) = Schedule((double)n / steps);
        return (new Tensor<double>(alpha, new[] { steps + 1 }).Cast<T>().ToArray(),
            new Tensor<double>(sigma, new[] { steps + 1 }).Cast<T>().ToArray());
    }

    /// <summary>
    /// The teacher's DDIM sampling (Eq. 5): from standard-normal noise <c>[K, d]</c> at τ = 1, for n = L … 1 with
    /// τ_n = n / L, the guided velocity prediction v̂ gives x̂₀ = α_n h − σ_n v̂ and ε̂ = σ_n h + α_n v̂, and
    /// h ← α_{n−1} x̂₀ + σ_{n−1} ε̂ (the angular schedule α = cos(πτ/2), σ = sin(πτ/2)).
    /// </summary>
    internal Tensor<T> TeacherSample(Tensor<T> noise, Tensor<T> text, Tensor<T> promptMel, double guidance, bool training, Random random)
    {
        int steps = _options.SamplingSteps;
        if (steps < 1) return noise;
        var (alphas, sigmas) = AngularSchedule(steps);
        var h = noise;
        for (int n = steps; n >= 1; n--)
        {
            var v = TeacherNetwork.Guided(h, Schedule((double)n / steps).Sigma, text, promptMel, guidance, training, random);
            var x0 = Engine.TensorSubtract(Engine.TensorMultiplyScalar(h, alphas[n]), Engine.TensorMultiplyScalar(v, sigmas[n]));
            var eps = Engine.TensorAdd(Engine.TensorMultiplyScalar(h, sigmas[n]), Engine.TensorMultiplyScalar(v, alphas[n]));
            h = Engine.TensorAdd(Engine.TensorMultiplyScalar(x0, alphas[n - 1]), Engine.TensorMultiplyScalar(eps, sigmas[n - 1]));
        }
        return h;
    }

    /// <summary>The distilled student's one-step map H(ξ; ω, x′, t) (its level input is the guidance scale).</summary>
    internal Tensor<T> StudentSample(Tensor<T> noise, Tensor<T> text, Tensor<T> promptMel, double guidance, bool training, Random random) =>
        StudentNetwork.Forward(noise, guidance, text, promptMel, false, training, random);

    // ---------------------------------------------------------------- synthesis

    /// <summary>
    /// The waveform of phonemes in the prompt's voice: the prompt-text encoder's aligned text embeddings and global style;
    /// the time-varying style sampled from noise <c>[K, d]</c> (the student, or the teacher's guided DDIM); durations from
    /// the duration decoder, then pitch and energy at the mel rate from the pitch-energy decoder; StyleTTS 2's decoder.
    /// </summary>
    internal Tensor<T> SynthesizeWave(int[] ids, Tensor<T> promptMel, Tensor<T> noise, Random random)
    {
        var o = _options;
        var (text, style) = PromptTextNetwork.Forward(ids, promptMel, false, random);
        var e = BertEncoderLayer.Forward(BertNetwork.Encode(ids, false, random));
        var h = o.UseDistilledSampler ? StudentSample(noise, e, promptMel, o.Guidance, false, random)
            : TeacherSample(noise, e, promptMel, o.Guidance, false, random);
        var frames = FramesOf(DurationDecoderNetwork.Forward(e, h, false, random));
        var phonemeOfFrame = PhonemeOfFrame(frames, 2);
        var prosody = ProsodyDecoderNetwork.Forward(Upsample(e, phonemeOfFrame), h, false, random);          // [2T, 2]
        int melFrames = phonemeOfFrame.Length;
        var pitch = Engine.Reshape(Engine.TensorSlice(prosody, new[] { 0, 0 }, new[] { melFrames, 1 }), new[] { 1, melFrames });
        var energy = Engine.Reshape(Engine.TensorSlice(prosody, new[] { 0, 1 }, new[] { melFrames, 1 }), new[] { 1, melFrames });
        var asr = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(text), AlignmentOf(frames)), new[] { 1, o.ModelDim, frames.Sum() });
        return DecoderNetwork.Forward(asr, pitch, energy, style, false, random);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the phoneme symbol ids; the output is the waveform in the voice's speaker.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var ids = ToIds(input, nameof(input));
            var prompt = MelsFirst(RequireVoice().Reference!);
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
            return SynthesizeWave(ids, prompt, Gaussian(new[] { _options.StyleLength, _options.ModelDim }, random), random);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = NumOps.FromDouble(StyleTts2StyleDiffusion<T>.Gaussian(random));
        return t;
    }

    // ---------------------------------------------------------------- pretrained components

    /// <summary>Loads the released PL-BERT (yl4579/PL-BERT's <c>step_1000000.t7</c>, the <c>module.encoder.</c> prefix
    /// removed).</summary>
    public void LoadBertWeights(string path)
    {
        using var file = TorchCheckpointFile.Open(path);
        BertNetwork.LoadTorchWeights(StyleTTS<T>.Reader(file, "module.encoder.", "encoder."));
    }

    /// <summary>Loads StyleTTS 2's released text aligner (<c>Utils/ASR/epoch_00080.pth</c>'s <c>model</c> entry).</summary>
    public void LoadAlignerWeights(string path)
    {
        using var file = TorchCheckpointFile.Open(path);
        AlignerNetwork.Load(StyleTTS<T>.StripPrefix(StyleTTS<T>.Reader(file, "model."), "text_aligner."), "text_aligner");
    }

    /// <summary>Loads StyleTTS 2's released pitch extractor (<c>Utils/JDC/bst.t7</c>'s <c>net</c> entry).</summary>
    public void LoadPitchExtractorWeights(string path)
    {
        using var file = TorchCheckpointFile.Open(path);
        PitchExtractorNetwork.Load(StyleTTS<T>.StripPrefix(StyleTTS<T>.Reader(file, "net."), "pitch_extractor."), "pitch_extractor");
    }

    /// <summary>Loads WavLM-base-plus (a transformers <c>WavLMModel</c> state dict).</summary>
    public void LoadSlmWeights(string path)
    {
        using var file = TorchCheckpointFile.Open(path);
        SlmNetwork.LoadTorchWeights(StyleTTS<T>.Reader(file, "wavlm."), name => file.Contains(name) || file.Contains("wavlm." + name));
    }

    /// <summary>Loads a WeSpeaker ResNet34 (<c>avg_model.pt</c>, e.g. its VoxCeleb model).</summary>
    public void LoadSpeakerWeights(string path)
    {
        using var file = TorchCheckpointFile.Open(path);
        SpeakerNetwork.LoadTorchWeights(StyleTTS<T>.Reader(file, "model.", "module."));
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "StyleTTSZS-Native" : "StyleTTSZS-ONNX",
            Description = "StyleTTS-ZS: Efficient High-Quality Zero-Shot Text-to-Speech Synthesis with Distilled Time-Varying Style Diffusion (Li et al., 2024)",
            FeatureCount = _options.ModelDim,
        };
        metadata.AdditionalInfo["Architecture"] = "StyleTTSZS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(StyleTTSZS<T>));
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
