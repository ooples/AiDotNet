using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// StyleTTS 2: text-to-speech through style diffusion and adversarial training with large speech language models
/// (Li et al. 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The paper's modules (§3; reference yl4579/StyleTTS2 <c>models.py</c>): StyleTTS's text encoder and two of its style
/// encoders, one for the acoustic style the decoder reads and one for the prosodic style the predictors read; PL-BERT, a
/// phoneme-level ALBERT, whose features feed the prosody predictor (through a linear map) and condition the style
/// diffusion; a prosody predictor whose duration head predicts 50 bins whose sigmoids sum to the duration; a style
/// diffusion model — a transformer denoiser with EDM preconditioning, sampled with ancestral DPM-2 — that samples both
/// styles from the text (and, for the multispeaker model, a reference style); and a decoder that ends in a HiFi-GAN or
/// iSTFTNet waveform generator with a harmonic-plus-noise source from the predicted pitch. Training also uses the
/// pre-trained text aligner and pitch extractor, multi-period and multi-resolution discriminators, and a WavLM
/// discriminator.
/// </para>
/// <para>
/// Synthesis follows the released inference notebooks: the sampled styles are mixed with the reference's
/// (<see cref="StyleTTS2Options.Alpha"/>, <see cref="StyleTTS2Options.Beta"/>) and the waveform is returned directly.
/// </para>
/// <para><b>For Beginners:</b> StyleTTS 2 reads text and speaks it, imagining a fitting speaking style for the text with
/// a diffusion model; given a short recording, it leans that style toward the recorded speaker.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models",
    "https://arxiv.org/abs/2306.07691",
    Year = 2023,
    Authors = "Li, Han, Raghavan, Mischler and Mesgarani"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.0, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 1e-4,
    Component = "generator", Provenance = RecipeProvenance.Stated,
    Source = "Section 4: AdamW with beta1 = 0, beta2 = 0.99, weight decay 1e-4 and learning rate 1e-4 (PL-BERT and, in "
             + "joint training, the acoustic modules at 1e-5); epsilon 1e-9 is the released code's (optimizers.py), whose "
             + "one-cycle schedule with div and final-div factors of 1 holds the rate constant.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.0, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 1e-4,
    Component = "discriminator", Provenance = RecipeProvenance.Stated,
    Source = "Section 4 (the discriminators train with the same AdamW).")]
public partial class StyleTTS2<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly StyleTTS2Options _options;
    private readonly bool _useNativeMode;
    private readonly Random _trainingRandom;
    private bool _disposed;

    private StyleTtsTextEncoder<T>? _textEncoder;
    private StyleTtsStyleEncoder<T>? _styleEncoder;
    private StyleTtsStyleEncoder<T>? _predictorEncoder;
    private StyleTtsProsodyPredictor<T>? _predictor;
    private StyleTts2Decoder<T>? _decoder;
    private AlbertNetwork<T>? _bert;
    private DenseLayer<T>? _bertEncoder;
    private StyleTts2StyleDiffusion<T>? _diffusion;
    private StyleTtsTextAligner<T>? _aligner;
    private StyleTtsPitchExtractor<T>? _pitch;
    private readonly List<LayerBase<T>> _textEncoderLayers = new();
    private readonly List<LayerBase<T>> _styleLayers = new();
    private readonly List<LayerBase<T>> _predictorEncoderLayers = new();
    private readonly List<LayerBase<T>> _predictorLayers = new();
    private readonly List<LayerBase<T>> _decoderLayers = new();
    private readonly List<LayerBase<T>> _bertLayers = new();
    private readonly List<LayerBase<T>> _bertEncoderLayers = new();
    private readonly List<LayerBase<T>> _diffusionLayers = new();
    private readonly List<LayerBase<T>> _alignerLayers = new();
    private readonly List<LayerBase<T>> _pitchLayers = new();
    private StyleTtsFrontEnd<T>? _frontEnd;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed StyleTTS 2 for inference.</summary>
    public StyleTTS2(NeuralNetworkArchitecture<T> architecture, string modelPath, StyleTTS2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new StyleTTS2Options();
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

    /// <summary>Creates a native, trainable StyleTTS 2.</summary>
    public StyleTTS2(NeuralNetworkArchitecture<T> architecture, StyleTTS2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new StyleTTS2Options();
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
        if (o.HiddenDim % 2 != 0) throw new ArgumentException("StyleTTS 2's BiLSTMs split the hidden width in half; use an even HiddenDim.");
        if (o.WindowLength > o.FftSize) throw new ArgumentException("The analysis window must fit the FFT.");
        if (o.UpsampleRates.Length == 0 || o.UpsampleRates.Length != o.UpsampleKernels.Length)
            throw new ArgumentException("Give one upsampling kernel per upsampling rate.");
        if (o.ResBlockKernels.Length != o.ResBlockDilations.Length)
            throw new ArgumentException("Give one dilation list per residual-block kernel.");
        int hop = o.UpsampleRates.Aggregate(1, (a, b) => a * b) * (o.DecoderType == StyleTts2DecoderType.IstftNet ? o.IstftHopSize : 1);
        if (2 * hop != o.HopSize)
            throw new ArgumentException($"The generator's hop ({hop}) must be half the mel hop ({o.HopSize}): the decoder upsamples the frames by 2.");
        if (o.UpsampleInitialChannels >> o.UpsampleRates.Length < 1)
            throw new ArgumentException("Too many upsampling stages for the initial channel count.");
        if (o.Bert.VocabSize < o.VocabSize)
            throw new ArgumentException("PL-BERT must embed every phoneme symbol.");
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int HiddenDim => _options.HiddenDim;

    /// <inheritdoc />
    /// <remarks>StyleTTS 2 has no flow; this is the number of style-diffusion sampling steps.</remarks>
    public int NumFlowSteps => _options.DiffusionSteps;

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice => _options.Multispeaker ? TtsSupervision.SpeakerReference : TtsSupervision.None;

    /// <inheritdoc />
    /// <remarks>Its synthesized length follows its predicted durations.</remarks>
    public override bool TrainsOnSampleObjective => true;

    private bool HasPaperLayers => _textEncoder is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    // The modules, for tests and training.
    internal StyleTtsTextEncoder<T> TextEncoderNetwork => _textEncoder ?? throw NotNative();
    internal StyleTtsStyleEncoder<T> StyleEncoderNetwork => _styleEncoder ?? throw NotNative();
    internal StyleTtsStyleEncoder<T> PredictorEncoderNetwork => _predictorEncoder ?? throw NotNative();
    internal StyleTtsProsodyPredictor<T> PredictorNetwork => _predictor ?? throw NotNative();
    internal StyleTts2Decoder<T> DecoderNetwork => _decoder ?? throw NotNative();
    internal AlbertNetwork<T> BertNetwork => _bert ?? throw NotNative();
    internal DenseLayer<T> BertEncoderLayer => _bertEncoder ?? throw NotNative();
    internal StyleTts2StyleDiffusion<T> DiffusionNetwork => _diffusion ?? throw NotNative();
    internal StyleTtsTextAligner<T> AlignerNetwork => _aligner ?? throw NotNative();
    internal StyleTtsPitchExtractor<T> PitchExtractorNetwork => _pitch ?? throw NotNative();

    /// <summary>The symbol table and mel spectrogram StyleTTS 2 shares with StyleTTS.</summary>
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
        _textEncoder = new StyleTtsTextEncoder<T>(Engine, _textEncoderLayers, o.HiddenDim, o.TextKernel, o.TextLayers, o.VocabSize);
        _styleEncoder = new StyleTtsStyleEncoder<T>(Engine, _styleLayers, o.StyleEncoderDim, o.StyleDim, o.MaxConvDim);
        _predictorEncoder = new StyleTtsStyleEncoder<T>(Engine, _predictorEncoderLayers, o.StyleEncoderDim, o.StyleDim, o.MaxConvDim);
        _predictor = new StyleTtsProsodyPredictor<T>(Engine, _predictorLayers, o.StyleDim, o.HiddenDim, o.TextLayers, o.DropoutRate, o.DurationBins);
        _decoder = new StyleTts2Decoder<T>(Engine, _decoderLayers, o.DecoderType, o.HiddenDim, o.StyleDim, o.DecoderBottleneck, o.ResidualDim,
            o.ResBlockKernels, o.UpsampleRates, o.UpsampleInitialChannels, o.ResBlockDilations, o.UpsampleKernels, o.IstftFftSize,
            o.IstftHopSize, o.SampleRate);
        _bert = new AlbertNetwork<T>(Engine, _bertLayers, o.Bert);
        _bertEncoder = StyleTtsOps.Own(_bertEncoderLayers, StyleTtsOps.Linear<T>(o.HiddenDim));
        var denoiser = new StyleTts2StyleDenoiser<T>(Engine, _diffusionLayers, 2 * o.StyleDim, o.Bert.Hidden, o.DiffusionLayers,
            o.DiffusionHeads, o.DiffusionHeadFeatures, o.DiffusionMultiplier, o.Multispeaker ? 2 * o.StyleDim : null, o.DiffusionMaxLength, random);
        _diffusion = new StyleTts2StyleDiffusion<T>(Engine, denoiser, o.SigmaData, o.LogSigmaMean, o.LogSigmaStd);
        _aligner = new StyleTtsTextAligner<T>(Engine, _alignerLayers, o.MelChannels, o.AlignerHidden, o.VocabSize, o.AlignerTokenEmbedding, o.AlignerConvLayers);
        _pitch = new StyleTtsPitchExtractor<T>(Engine, _pitchLayers, o.PitchExtractorChannels, o.PitchExtractorLstm, o.MelChannels);
        InitializeDiscriminators();
        // The duration head is LinearNorm: Xavier-uniform weights (gain 1).
        double bound = Math.Sqrt(6.0 / (o.HiddenDim + o.DurationBins));
        TorchParameters.Linear(Engine, _predictor.DurationProjection, o.HiddenDim, o.DurationBins,
            Enumerable.Range(0, o.HiddenDim * o.DurationBins).Select(_ => (2 * random.NextDouble() - 1) * bound).ToArray(), new double[o.DurationBins]);

        MaterializeModules();

        AddEncoderDecoderLayers(_textEncoderLayers.Cast<ILayer<T>>().ToList(), _decoderLayers.Cast<ILayer<T>>().ToList());
        foreach (var group in new[] { _styleLayers, _predictorEncoderLayers, _predictorLayers, _bertLayers, _bertEncoderLayers, _diffusionLayers,
                     _alignerLayers, _pitchLayers })
            ComponentLayers.AddRange(group);
        AddDiscriminatorLayers();
    }

    private IEnumerable<LayerBase<T>> GeneratorLayers =>
        _textEncoderLayers.Concat(_styleLayers).Concat(_predictorEncoderLayers).Concat(_predictorLayers).Concat(_decoderLayers)
            .Concat(_bertLayers).Concat(_bertEncoderLayers).Concat(_diffusionLayers).Concat(_alignerLayers).Concat(_pitchLayers);

    /// <summary>
    /// Runs every module once on placeholder inputs, in inference mode and outside the tape, so the dense layers that
    /// size themselves on their first forward hold their weights from construction.
    /// </summary>
    private void MaterializeModules()
    {
        var o = _options;
        var all = GeneratorLayers.Concat(DiscriminatorLayerList).ToList();
        foreach (var layer in all) layer.SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = new Random(0);
            int frames = 80, tokens = 2;
            var mel = new Tensor<T>(new[] { o.MelChannels, frames });
            var ids = new[] { 0, 0 };
            var text = _textEncoder!.Forward(ToTensor(ids), false, random);
            var acoustic = _styleEncoder!.Forward(mel);
            var prosodic = _predictorEncoder!.Forward(mel);
            var bert = _bert!.Encode(ids, false, random);
            var dEn = _bertEncoder!.Forward(bert);
            var reference = Engine.TensorConcatenate(new[] { acoustic, prosodic }, 1);
            _diffusion!.Denoise(reference, 1.0, bert, o.Multispeaker ? reference : null, 1.0);
            var encoded = _predictor!.EncodeText(Engine.Reshape(Engine.TensorTranspose(dEn), new[] { 1, o.HiddenDim, tokens }), prosodic, false, random);
            _predictor.Durations(encoded, false, random);
            int half = 4;
            var aligned = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(encoded), new Tensor<T>(new[] { tokens, half })),
                new[] { 1, o.HiddenDim + o.StyleDim, half });
            var (f0, energy) = _predictor.Prosody(aligned, prosodic, false, random);
            _decoder!.Forward(new Tensor<T>(new[] { 1, o.HiddenDim, half }), f0, energy, acoustic, false, random);
            _aligner!.Decode(mel, ids, false, random, 0.0);
            _aligner.CtcLogits(mel, false, random);
            _pitch!.Forward(mel, false, random);
            MaterializeDiscriminators();
        }
        finally
        {
            foreach (var layer in all) layer.SetTrainingMode(true);
        }
    }

    // ---------------------------------------------------------------- front end

    /// <summary>The symbol ids of IPA phonemes as espeak writes them (words separated by spaces), preceded by the pad
    /// symbol as the reference's inference prepends it; characters outside the table are skipped.</summary>
    public int[] EncodePhonemes(string phonemes)
    {
        var framed = StyleTtsFrontEnd<T>.EncodePhonemes(phonemes);
        return framed.Take(framed.Length - 1).ToArray();
    }

    /// <inheritdoc />
    /// <remarks>The text's espeak-style IPA phonemes with stress (<see cref="EnglishG2P"/>), as symbol ids preceded by the
    /// pad symbol.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var builder = new System.Text.StringBuilder();
        foreach (var symbol in EnglishG2P.Default.Phonemize(text))
            builder.Append(symbol == EnglishG2P.WordSeparator ? " " : symbol);
        var ids = EncodePhonemes(builder.ToString());
        if (ids.Length <= 1) throw new ArgumentException("The text has no pronounceable content.", nameof(text));
        return ToTensor(ids.Take(Math.Min(_options.MaxTextLength, Math.Min(_options.Bert.MaxPositions, _options.DiffusionMaxLength))).ToArray());
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
        int limit = Math.Min(_options.Bert.MaxPositions, _options.DiffusionMaxLength);
        if (ids.Length > limit)
            throw new ArgumentException($"PL-BERT and the style denoiser read at most {limit} symbols.", name);
        return ids;
    }

    /// <summary>The model's normalized log-mel spectrogram <c>[mels, frames]</c> of a waveform at its sample rate (as
    /// StyleTTS computes it).</summary>
    public Tensor<T> ComputeMel(Tensor<T> audio) => FrontEnd.ComputeMel(audio);

    /// <summary>A voice for <see cref="TtsModelBase{T}.Voice"/>: a reference recording at the model's sample rate (the
    /// reference notebooks trim its silences first) whose speaker the synthesized speech takes.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording)
    {
        var mel = ComputeMel(recording ?? throw new ArgumentNullException(nameof(recording)));
        return new TtsVoice<T> { Reference = Engine.TensorTranspose(mel), ReferenceAudio = recording };
    }

    /// <summary>The reference style <c>[1, 2 · style]</c> — acoustic then prosodic — of a mel spectrogram
    /// <c>[frames, mels]</c> (the notebooks' <c>compute_style</c>).</summary>
    public Tensor<T> ComputeStyle(Tensor<T> referenceMel)
    {
        if (referenceMel is null) throw new ArgumentNullException(nameof(referenceMel));
        using var _ = new NoGradScope<T>();
        return ReferenceStyle(MelsFirst(referenceMel));
    }

    private Tensor<T> ReferenceStyle(Tensor<T> mel) =>
        Engine.TensorConcatenate(new[] { StyleEncoderNetwork.Forward(mel), PredictorEncoderNetwork.Forward(mel) }, 1);

    private Tensor<T> MelsFirst(Tensor<T> mel)
    {
        if (mel.Rank != 2 || mel.Shape[1] != _options.MelChannels)
            throw new ArgumentException($"Expected a mel spectrogram [frames, {_options.MelChannels}].", nameof(mel));
        return Engine.TensorTranspose(mel);
    }

    // ---------------------------------------------------------------- synthesis

    private Tensor<T> AlignmentOf(int[] durations)
    {
        int total = durations.Sum();
        var alignment = new Tensor<T>(new[] { durations.Length, total });
        int frame = 0;
        for (int i = 0; i < durations.Length; i++)
            for (int f = 0; f < durations[i]; f++) alignment[i, frame++] = NumOps.One;
        return alignment;
    }

    /// <summary>The frames of each phoneme from the duration head's logits <c>[L, bins]</c>: the rounded (half to even)
    /// sum of the bins' sigmoids, at least 1.</summary>
    internal int[] FramesOf(Tensor<T> logits)
    {
        int length = logits.Shape[0], bins = logits.Shape[1];
        var frames = new int[length];
        for (int i = 0; i < length; i++)
        {
            double sum = 0;
            for (int b = 0; b < bins; b++) sum += 1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(logits[i, b])));
            frames[i] = Math.Max(1, (int)Math.Round(sum, MidpointRounding.ToEven));
        }
        frames[length - 1] += _options.FinalPhonemeExtraFrames;
        return frames;
    }

    // HiFi-GAN decoders read the aligned features one frame late at inference (the LibriTTS notebook).
    private Tensor<T> ShiftOneFrame(Tensor<T> x)
    {
        if (_options.DecoderType != StyleTts2DecoderType.HiFiGan) return x;
        int frames = x.Shape[2];
        var first = Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, x.Shape[1], 1 });
        return frames == 1 ? x : Engine.TensorConcatenate(new[] { first, Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, x.Shape[1], frames - 1 }) }, 2);
    }

    /// <summary>
    /// The waveform of phonemes (the reference notebooks' <c>inference</c>): PL-BERT features; both styles sampled by the
    /// style diffusion from <paramref name="noise"/> <c>[1, 2 · style]</c> and, for the multispeaker model, mixed with the
    /// reference style; durations, pitch and energy from the predictor; the decoder. <paramref name="stepNoise"/> supplies
    /// the sampler's per-step draws (null: drawn from <paramref name="random"/>).
    /// </summary>
    internal Tensor<T> SynthesizeWave(int[] ids, Tensor<T>? referenceStyle, Tensor<T> noise, Random random, Func<int, Tensor<T>>? stepNoise = null)
    {
        var o = _options;
        int length = ids.Length, s = o.StyleDim;
        if (o.Multispeaker && referenceStyle is null) throw new ArgumentNullException(nameof(referenceStyle), "The multispeaker model reads a reference style.");
        var text = TextEncoderNetwork.Forward(ToTensor(ids), false, random);                                  // [1, C, L]
        var bert = BertNetwork.Encode(ids, false, random);                                                    // [L, 768]
        var dEn = Engine.Reshape(Engine.TensorTranspose(BertEncoderLayer.Forward(bert)), new[] { 1, o.HiddenDim, length });
        var sampled = DiffusionNetwork.Sample(noise, bert, o.Multispeaker ? referenceStyle : null, o.DiffusionSteps, o.EmbeddingScale, random, stepNoise);
        var acoustic = Engine.TensorSlice(sampled, new[] { 0, 0 }, new[] { 1, s });
        var prosodic = Engine.TensorSlice(sampled, new[] { 0, s }, new[] { 1, s });
        if (o.Multispeaker)
        {
            Tensor<T> Mix(Tensor<T> a, double w, int offset) => Engine.TensorAdd(Engine.TensorMultiplyScalar(a, NumOps.FromDouble(w)),
                Engine.TensorMultiplyScalar(Engine.TensorSlice(referenceStyle!, new[] { 0, offset }, new[] { 1, s }), NumOps.FromDouble(1 - w)));
            acoustic = Mix(acoustic, o.Alpha, 0);
            prosodic = Mix(prosodic, o.Beta, s);
        }
        var encoded = PredictorNetwork.EncodeText(dEn, prosodic, false, random);                              // [L, C + S]
        var frames = FramesOf(PredictorNetwork.Durations(encoded, false, random));
        var alignment = AlignmentOf(frames);
        int total = frames.Sum();
        var en = ShiftOneFrame(Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(encoded), alignment), new[] { 1, o.HiddenDim + s, total }));
        var (f0, energy) = PredictorNetwork.Prosody(en, prosodic, false, random);
        var asr = ShiftOneFrame(Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(text, new[] { o.HiddenDim, length }), alignment),
            new[] { 1, o.HiddenDim, total }));
        var wave = DecoderNetwork.Forward(asr, f0, energy, acoustic, false, random);
        int keep = Math.Max(0, wave.Length - o.TrimEndSamples);
        return keep == wave.Length ? wave : Engine.TensorSlice(wave, new[] { 0 }, new[] { keep });
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the phoneme symbol ids (<see cref="PreprocessText"/>); the output is the
    /// waveform at the model's sample rate.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var ids = ToIds(input, nameof(input));
            Tensor<T>? reference = _options.Multispeaker ? ReferenceStyle(MelsFirst(RequireVoice().Reference!)) : null;
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
            var noise = new Tensor<T>(new[] { 1, 2 * _options.StyleDim });
            for (int i = 0; i < noise.Length; i++) noise[i] = NumOps.FromDouble(StyleTts2StyleDiffusion<T>.Gaussian(random));
            return SynthesizeWave(ids, reference, noise, random);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>
    /// Loads a StyleTTS 2 checkpoint flattened to named tensors (safetensors, or a <c>.pt</c> state dict) whose names are
    /// the reference's module keys (<c>bert</c>, <c>bert_encoder</c>, <c>text_encoder</c>, <c>style_encoder</c>,
    /// <c>predictor_encoder</c>, <c>predictor</c>, <c>decoder</c>, <c>diffusion</c>, <c>text_aligner</c>,
    /// <c>pitch_extractor</c> and optionally <c>mpd</c>, <c>msd</c>, <c>wd</c>) and their parameters, with or without the
    /// checkpoint's <c>net.</c> prefix.
    /// </summary>
    public void LoadReferenceWeights(string path, bool includeDiscriminators = false)
    {
        IReadOnlyDictionary<string, double[]>? waveState = null;
        if (includeDiscriminators)
        {
            // The mpd and msd tensors, keyed as the reference names them (prefixes net. and module. removed).
            var state = new Dictionary<string, double[]>();
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            {
                var name = entry.Key.StartsWith("net.", StringComparison.Ordinal) ? entry.Key.Substring(4) : entry.Key;
                name = name.Replace(".module.", ".");
                if (name.StartsWith("mpd.", StringComparison.Ordinal) || name.StartsWith("msd.", StringComparison.Ordinal))
                    state[name] = entry.Value.ToVector().Select(v => (double)v).ToArray();
            }
            waveState = state;
        }
        LoadReferenceWeights(StyleTTS<T>.Reader(path, "net."), waveState);
    }

    /// <summary>Loads the generator's modules from <paramref name="read"/> and, given their state, the discriminators.</summary>
    internal void LoadReferenceWeights(TorchReader read, IReadOnlyDictionary<string, double[]>? waveDiscriminatorState = null)
    {
        var o = _options;
        BertNetwork.LoadTorchWeights((name, shape) => read(name, shape), "bert.");
        StyleTtsOps.LoadLinear(Engine, BertEncoderLayer, read, "bert_encoder", o.Bert.Hidden, o.HiddenDim);
        TextEncoderNetwork.Load(read, "text_encoder");
        StyleEncoderNetwork.Load(Engine, read, "style_encoder");
        PredictorEncoderNetwork.Load(Engine, read, "predictor_encoder");
        PredictorNetwork.Load(read, "predictor");
        DecoderNetwork.Load(read, "decoder");
        DiffusionNetwork.Denoiser.Load(read, "diffusion.diffusion.net");
        AlignerNetwork.Load(read, "text_aligner");
        PitchExtractorNetwork.Load(read, "pitch_extractor");
        if (waveDiscriminatorState is not null) LoadDiscriminators(read, waveDiscriminatorState);
    }

    /// <summary>Loads the released PL-BERT (<c>Utils/PLBERT/step_1000000.t7</c>'s <c>net</c> entry with the
    /// <c>module.encoder.</c> prefix removed, as <c>load_plbert</c> does, saved as safetensors or <c>.pt</c>).</summary>
    public void LoadBertWeights(string path) =>
        BertNetwork.LoadTorchWeights((name, shape) => StyleTTS<T>.Reader(path, "module.encoder.", "encoder.")(name, shape));

    /// <summary>Loads the released text aligner (<c>Utils/ASR/epoch_00080.pth</c>'s <c>model</c> entry).</summary>
    public void LoadAlignerWeights(string path) =>
        AlignerNetwork.Load(StyleTTS<T>.StripPrefix(StyleTTS<T>.Reader(path, "model."), "text_aligner."), "text_aligner");

    /// <summary>Loads the released pitch extractor (<c>Utils/JDC/bst.t7</c>'s <c>net</c> entry).</summary>
    public void LoadPitchExtractorWeights(string path) =>
        PitchExtractorNetwork.Load(StyleTTS<T>.StripPrefix(StyleTTS<T>.Reader(path, "net."), "pitch_extractor."), "pitch_extractor");

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "StyleTTS2-Native" : "StyleTTS2-ONNX",
            Description = "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models (Li et al., 2023)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.TextLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "StyleTTS2";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Decoder"] = _options.DecoderType.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(StyleTTS2<T>));
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
