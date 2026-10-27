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

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// Mega-TTS 2 (Jiang et al. 2024): zero-shot speech from prompts of any length. An acoustic autoencoder splits speech into
/// content (a Transformer over the phonemes), prosody (a VQ encoder that pools the mel spectrogram eight times into
/// codes) and timbre (the multi-reference timbre encoder, whose content attends to the joined prompt recordings); a
/// prosody latent language model (P-LLM) continues the prompts' prosody codes for the new text, an autoregressive duration
/// model their phoneme durations, and HiFi-GAN V1 makes the audio. Prosody interpolation mixes the P-LLM's distributions
/// for two speakers' prosody prompts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The paper's model with its Table 5 sizes. Mega-TTS 2 has no released code; what the paper leaves unstated comes
/// from the authors' code: NATSpeech's encoder layers and convolution stacks, SyntaSpeech's multi-length discriminator, and
/// the decoder-only Transformer of MegaTTS 3's released duration predictor (whose code embedding of K + 2 entries is the
/// paper's 1026), and from CVQ-VAE's code for its quantizer.</para>
/// <para><b>For Beginners:</b> give Mega-TTS 2 one or more recordings of someone speaking (<see cref="TtsModelBase{T}.Voice"/>,
/// or <see cref="PromptSentences"/> for several), each with its phonemes and their durations, and it says new text in that
/// voice and with that person's way of speaking; the more sentences, the closer it gets. Set <see cref="ProsodyPrompts"/>
/// to recordings of another speaker to borrow their speaking style while keeping the voice. Training goes in two stages:
/// the acoustic model, then the language models (<see cref="MegaTTS2Options.Stage"/>).</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new MegaTTS2&lt;float&gt;(architecture, new MegaTTS2Options());
/// model.LoadVocoderWeights("generator_v1");   // a HiFi-GAN V1 generator trained on this model's mel spectrograms
/// model.PromptSentences = new[]
/// {
///     new TtsVoice&lt;float&gt; { ReferenceAudio = clip1, ReferenceTokens = model.TextToTokens(text1), ReferenceDurations = durations1 },
///     new TtsVoice&lt;float&gt; { ReferenceAudio = clip2, ReferenceTokens = model.TextToTokens(text2), ReferenceDurations = durations2 },
/// };
/// var audio = model.Synthesize("Hello, how are you doing today?");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Mega-TTS 2: Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis",
    "https://arxiv.org/abs/2307.07218",
    Year = 2024,
    Authors = "Jiang et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-9, MaxGradientNorm = 1.0,
    Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000, Component = "acoustic",
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Jiang et al. 2024, Sec. 4.1: Adam with beta1 0.9, beta2 0.999 and epsilon 1e-9 on the learning rate schedule "
             + "of Vaswani et al. 2017; the warmup of 4000 steps is that paper's, and the gradient norm clip of 1 is the "
             + "authors' NATSpeech code base (clip_grad_norm). 600K steps at 48 sentences per GPU on 4 GPUs.")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-9, MaxGradientNorm = 1.0,
    Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000, Component = "prosody",
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Jiang et al. 2024, Sec. 4.1: the P-LLM trains with the same optimizer and schedule for 300K steps, 4000 tokens per GPU on 8 GPUs.")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-9, MaxGradientNorm = 1.0,
    Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000, Component = "duration",
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Jiang et al. 2024, Sec. 4.1: the duration model trains with the P-LLM in the second stage.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-6, WeightDecay = 0.0,
    MaxGradientNorm = 1.0, Component = "discriminator", Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Jiang et al. 2024 follows HiFiSinger's multi-length discriminator as SyntaSpeech builds it; SyntaSpeech's "
             + "synta.yaml trains it with AdamW at 1e-4 (betas 0.9, 0.98, eps 1e-6, weight decay 0), halved every 40000 steps.")]
public partial class MegaTTS2<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly MegaTTS2Options _options;
    private readonly bool _useNativeMode;
    private readonly string[] _vocabulary;
    private readonly MegaTtsMel _mel;
    private bool _disposed;

    private MegaTts2Network<T>? _network;
    private HiFiGanGenerator<T>? _vocoder;

    private readonly List<LayerBase<T>> _acousticLayers = new();
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _durationLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private readonly List<LayerBase<T>> _vocoderLayers = new();

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed Mega-TTS 2 for inference.</summary>
    public MegaTTS2(NeuralNetworkArchitecture<T> architecture, string modelPath, MegaTTS2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new MegaTTS2Options();
        _useNativeMode = false;
        (_vocabulary, _mel) = ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Mega-TTS 2.</summary>
    public MegaTTS2(NeuralNetworkArchitecture<T> architecture, MegaTTS2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new MegaTTS2Options();
        _useNativeMode = true;
        (_vocabulary, _mel) = ConfigureBase();
        InitializeLayers();
    }

    private (string[] Vocabulary, MegaTtsMel Mel) ConfigureBase()
    {
        var o = _options;
        if (o.Language == MegaTtsLanguage.Chinese && o.PhoneSet is null)
            throw new ArgumentException("The Chinese front end's phones come with the data (NATSpeech's ph_set.json): set PhoneSet.");
        if (o.VocoderRates.Aggregate(1, (a, b) => a * b) != o.HopSize)
            throw new ArgumentException("The vocoder's upsampling rates must multiply to HopSize.");
        if (o.VocoderInitialChannels >> o.VocoderRates.Length < 1)
            throw new ArgumentException("Every vocoder stage halves the width: VocoderInitialChannels must keep at least one channel.");
        if (o.HiddenDim % o.ContentHeads != 0 || o.HiddenDim % o.TimbreHeads != 0) throw new ArgumentException("A width must divide by its heads.");
        if (o.LmHidden % o.LmHeads != 0 || (o.LmHidden / o.LmHeads) % 2 != 0 || o.DurationHidden % o.DurationHeads != 0 || (o.DurationHidden / o.DurationHeads) % 2 != 0)
            throw new ArgumentException("The language models' widths divide into heads of even width (rotary positions).");
        if (o.Compression < 1 || o.Downsampling < 1) throw new ArgumentException("The pooling rates are positive.");
        if (o.TopK < 1 || o.TopK > o.Codes) throw new ArgumentException("TopK lies in [1, Codes].");
        if (o.ProsodyInterpolation < 0 || o.ProsodyInterpolation > 1) throw new ArgumentException("ProsodyInterpolation lies in [0, 1].");
        if (o.CodeUsageDecay <= 0 || o.CodeUsageDecay >= 1) throw new ArgumentException("CodeUsageDecay lies in (0, 1).");
        if (o.DurationCodes < 1) throw new ArgumentException("DurationCodes is positive.");
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        return (o.Vocabulary, new MegaTtsMel(o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency));
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int HiddenDim => _options.HiddenDim;

    /// <inheritdoc />
    /// <remarks>Mega-TTS 2 decodes in one pass; it has no flow or diffusion steps.</remarks>
    public int NumFlowSteps => 0;

    /// <summary>The target speaker's prompt sentences, joined for the timbre encoder and read in order by the language
    /// models; null takes <see cref="TtsModelBase{T}.Voice"/> alone. Each is a recording
    /// (<see cref="TtsVoice{T}.ReferenceAudio"/>, or a mel spectrogram <see cref="TtsVoice{T}.Reference"/>) and, for the
    /// language models, its phonemes and their durations.</summary>
    public IReadOnlyList<TtsVoice<T>>? PromptSentences { get; set; }

    /// <summary>Another speaker's prompt sentences, whose prosody the P-LLM mixes in with weight
    /// <see cref="MegaTTS2Options.ProsodyInterpolation"/> (§3.4); each needs its phonemes and durations. Null samples the
    /// target speaker's prosody alone.</summary>
    public IReadOnlyList<TtsVoice<T>>? ProsodyPrompts { get; set; }

    /// <inheritdoc />
    /// <remarks>Mega-TTS 2 is zero-shot: it reads the voice's recording for the timbre, and with the recording's phonemes
    /// and their durations its prosody codes and durations as the language models' prompt.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    /// <remarks>The content is expanded to the frames by the phoneme durations, from a forced aligner (the paper uses the
    /// Montreal Forced Aligner). <see cref="TtsTrainingSample{T}.SpeakerReference"/> is the timbre encoder's reference: other
    /// recordings of the same speaker, joined (a waveform, or a mel spectrogram); without it the target itself is the
    /// reference.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    private bool HasPaperLayers => _network is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal MegaTts2Network<T> Network => _network ?? throw NotNative();
    internal HiFiGanGenerator<T> Vocoder => _vocoder ?? throw NotNative();
    internal IReadOnlyList<string> Vocabulary => _vocabulary;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_vocabulary.Length) : base.GetInputDomain(inputShape);

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
        _network = new MegaTts2Network<T>(Engine, _acousticLayers, _lmLayers, _durationLayers, _discriminatorLayers, new MegaTts2Configuration
        {
            PhoneVocabulary = _vocabulary.Length, Hidden = o.HiddenDim, ContentLayers = o.ContentLayers, ContentHeads = o.ContentHeads,
            ContentFilter = o.ContentFilter, ProsodyHidden = o.ProsodyHidden, ProsodyBlocks = o.ProsodyBlocks, Compression = o.Compression,
            Codes = o.Codes, CodeDimension = o.CodeDimension, Distance = o.CodeDistance, Anchor = o.CodeAnchor,
            CommitmentWeight = o.CommitmentWeight, UsageDecay = o.CodeUsageDecay, TimbreHidden = o.TimbreHidden, TimbreBlocks = o.TimbreBlocks,
            Downsampling = o.Downsampling, TimbreHeads = o.TimbreHeads, DecoderBlocks = o.DecoderBlocks, Mels = o.MelChannels,
            LmLayers = o.LmLayers, LmHidden = o.LmHidden, LmHeads = o.LmHeads, DurationLayers = o.DurationLayers,
            DurationHidden = o.DurationHidden, DurationHeads = o.DurationHeads, DurationCodes = o.DurationCodes,
            DiscriminatorWindows = o.DiscriminatorWindows, DiscriminatorHidden = o.DiscriminatorHidden,
        });
        // CVQ-VAE starts its codebook uniform in ±1/K.
        _network.Quantizer.Reset(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.TrainingSeed));
        _vocoder = new HiFiGanGenerator<T>(Engine, o.MelChannels, o.VocoderInitialChannels, o.VocoderRates, o.VocoderKernels,
            o.VocoderResBlockKernels, o.VocoderResBlockDilations, resblockType1: true);
        _vocoderLayers.AddRange(_vocoder.Layers);
        AddUnsplitLayers(_acousticLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_lmLayers);
        ComponentLayers.AddRange(_durationLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
        ComponentLayers.AddRange(_vocoderLayers);
    }

    // ---------------------------------------------------------------- text

    /// <inheritdoc />
    /// <remarks>The text's phone ids: NATSpeech's English processor (or the Chinese one), encoded over the phone set.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        if (text is null) throw new ArgumentNullException(nameof(text));
        var phones = _options.Language == MegaTtsLanguage.English ? NatSpeechFrontEnd.EnglishPhonesOf(text) : NatSpeechFrontEnd.ChinesePhonesOf(text);
        var ids = NatSpeechFrontEnd.Encode(phones, _vocabulary).Take(_options.MaxTextLength).ToArray();
        if (ids.Length == 0) throw new ArgumentException("The text has no phonemes.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    internal int[] Ids(Tensor<T> tokens, string name)
    {
        var ids = tokens.ToArray().Select(v => (int)Math.Round(NumOps.ToDouble(v))).ToArray();
        if (ids.Length == 0) throw new ArgumentException("There are no phonemes.", name);
        if (ids.Any(i => i < 0 || i >= _vocabulary.Length))
            throw new ArgumentOutOfRangeException(name, $"Phone ids lie in [0, {_vocabulary.Length}).");
        return ids;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        return base.Synthesize(text);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a text's phone ids (<see cref="PreprocessText"/>); the acoustic model's mel
    /// spectrogram <c>[frames, mels]</c> of them in the prompts' voice, which <see cref="Synthesize"/> turns into audio.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var ids = Ids(input, nameof(input));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return SynthesizeMel(ids, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    /// <remarks>HiFi-GAN V1 on the mel spectrogram.</remarks>
    protected override Tensor<T> PostprocessAudio(Tensor<T> output)
    {
        if (!HasPaperLayers || IsOnnxMode) return output;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Wave(output);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>A prompt sentence: its mel spectrogram, and with its phonemes and durations its content, frame map, phone
    /// durations, pooled content and prosody codes.</summary>
    internal sealed record Prompt(Tensor<T> Mel, int[]? Phones, int[]? Durations, Tensor<T>? Content, Tensor<T>? Pooled, int[]? Codes);

    internal Prompt ReadPrompt(TtsVoice<T> voice, string name)
    {
        var mel = voice.ReferenceAudio is not null ? MelOf(Doubles(voice.ReferenceAudio))
            : voice.Reference ?? throw new ArgumentException("A prompt is a recording (ReferenceAudio) or its mel spectrogram (Reference).", name);
        if (mel.Shape.Length != 2 || mel.Shape[1] != _options.MelChannels)
            throw new ArgumentException($"A prompt's mel spectrogram is [frames, {_options.MelChannels}].", name);
        if (voice.ReferenceTokens is null) return new Prompt(mel, null, null, null, null, null);
        var phones = Ids(voice.ReferenceTokens, name);
        var durations = voice.ReferenceDurations
            ?? throw new InvalidOperationException("A prompt's phonemes need their durations: set ReferenceDurations.");
        if (durations.Length != phones.Length) throw new ArgumentException("A prompt has one duration per phoneme.", name);
        var mel2ph = FrameMap(durations, mel.Shape[0]);
        var content = Content(phones);
        return new Prompt(mel, phones, durations, content, Network.PooledContent(content, mel2ph), Network.Quantizer.Encode(Network.ProsodyHidden(mel)));
    }

    /// <summary>
    /// The paper's zero-shot decoding: the target speaker's prompts joined for the MRTE, which the text's content attends
    /// to; the duration model's durations for the text after the prompts' durations, rounded; the P-LLM's prosody codes
    /// for the frames after the prompts' codes, top-k sampled (from the interpolated distribution with prosody prompts);
    /// the decoder's mel spectrogram <c>[frames, mels]</c>.
    /// </summary>
    internal Tensor<T> SynthesizeMel(int[] phones, Random random)
    {
        var target = (PromptSentences ?? new[] { RequireVoice() }).Select((v, i) => ReadPrompt(v, $"{nameof(PromptSentences)}[{i}]")).ToList();
        if (target.Count == 0) throw new InvalidOperationException("There are no prompt sentences.");
        var style = ProsodyPrompts?.Select((v, i) => ReadPrompt(v, $"{nameof(ProsodyPrompts)}[{i}]")).ToList();
        if (style is not null && style.Any(p => p.Codes is null))
            throw new InvalidOperationException("A prosody prompt needs its phonemes and durations.");
        var joined = target.Count == 1 ? target[0].Mel : Engine.TensorConcatenate(target.Select(p => p.Mel).ToArray(), 0);
        var content = Content(phones);
        var timbre = Network.Timbre(content, Network.TimbreStates(joined));
        var durations = PredictDurations(target, content, phones);
        var map = MegaTts3Aligner<T>.LengthRegulate(durations);
        if (map.Length == 0) return new Tensor<T>(new[] { 0, _options.MelChannels });
        var pooled = Network.PooledContent(content, map);
        var codes = SampleProsody(target, style, pooled, random);
        var prosody = Network.ProsodyAtFrames(Network.Quantizer.Lookup(codes), map.Length);
        return Network.DecodeMel(content, timbre, prosody, map);
    }

    /// <summary>The duration model's durations for the text's phonemes (padding at zero): teacher-forced on each prompt
    /// sentence's durations, then each prediction rounded (half to even) and read back as the next token. As the authors'
    /// released duration model infers (MegaTTS 3's <c>ARDurPredictor.infer</c>), every phoneme after the first lasts at
    /// least one frame.</summary>
    internal int[] PredictDurations(IReadOnlyList<Prompt> prompts, Tensor<T> content, int[] phones)
    {
        var model = Network.DurationModel;
        var cache = new MegaTts3DurationCache<T>();
        foreach (var p in prompts.Where(p => p.Durations is not null))
            model.Extend(DurationTokens(p.Durations!), Rows(p.Content!), cache);
        int l = phones.Length;
        var rows = Rows(content);
        var result = new int[l];
        int token = Network.DurationStart;
        for (int i = 0; i < l; i++)
        {
            var row = Engine.TensorSlice(rows, new[] { i, 0 }, new[] { 1, rows.Shape[1] });
            double predicted = NumOps.ToDouble(model.Extend(new[] { token }, row, cache)[0]);
            result[i] = phones[i] == 0 ? 0 : Math.Max(i == 0 ? 0 : 1, (int)Math.Round(predicted, MidpointRounding.ToEven));
            token = Math.Min(result[i], _options.DurationCodes);
        }
        return result;
    }

    /// <summary>A sentence's duration-model inputs: the start token, then each duration but the last as a token.</summary>
    internal int[] DurationTokens(int[] durations) =>
        new[] { Network.DurationStart }.Concat(durations.Take(durations.Length - 1).Select(d => Math.Min(d, _options.DurationCodes))).ToArray();

    /// <summary>A sentence's P-LLM inputs and features: the start token and each code as token code + 1, with the pooled
    /// content of the code each position predicts (zeros where it predicts the sentence's end).</summary>
    internal (int[] Tokens, Tensor<T> Features) ProsodyInputs(int[] codes, Tensor<T> pooled)
    {
        var tokens = new[] { Network.StartToken }.Concat(codes.Select(c => c + 1)).ToArray();
        var features = Engine.TensorConcatenate(new[] { pooled, new Tensor<T>(new[] { 1, pooled.Shape[1] }) }, 0);
        return (tokens, features);
    }

    /// <summary>
    /// The P-LLM's codes for the frames, one token at a time after the start token, each position reading its frames'
    /// pooled content; each prompt sentence read first in full. With prosody prompts, the distribution is
    /// (1 − γ)·p(·|target prompts) + γ·p(·|prosody prompts) (Eq. 3), the two sharing the sampled codes. The end token is
    /// not drawn (the length follows the durations); top-k sampling of what remains.
    /// </summary>
    internal int[] SampleProsody(IReadOnlyList<Prompt> target, IReadOnlyList<Prompt>? style, Tensor<T> pooled, Random random)
    {
        var model = Network.ProsodyModel;
        MegaTts3DurationCache<T> Prefill(IEnumerable<Prompt> prompts)
        {
            var cache = new MegaTts3DurationCache<T>();
            foreach (var p in prompts.Where(p => p.Codes is not null))
            {
                var (tokens, features) = ProsodyInputs(p.Codes!, p.Pooled!);
                model.Extend(tokens, features, cache);
            }
            return cache;
        }
        var targetCache = Prefill(target);
        var styleCache = style is null ? null : Prefill(style);
        double gamma = style is null ? 0 : _options.ProsodyInterpolation;
        int n = pooled.Shape[0], h = pooled.Shape[1], v = _options.Codes + 1;
        var codes = new int[n];
        int token = Network.StartToken;
        double[] Probabilities(MegaTts3DurationCache<T> cache, Tensor<T> row)
        {
            var logits = model.Extend(new[] { token }, row, cache);
            var p = new double[v];
            double max = double.NegativeInfinity;
            for (int c = 0; c < v; c++) max = Math.Max(max, p[c] = NumOps.ToDouble(logits[0, c]));
            double sum = 0;
            for (int c = 0; c < v; c++) sum += p[c] = Math.Exp(p[c] - max);
            for (int c = 0; c < v; c++) p[c] /= sum;
            return p;
        }
        for (int i = 0; i < n; i++)
        {
            var row = Engine.TensorSlice(pooled, new[] { i, 0 }, new[] { 1, h });
            var p = Probabilities(targetCache, row);
            if (styleCache is not null)
            {
                var q = Probabilities(styleCache, row);
                for (int c = 0; c < v; c++) p[c] = (1 - gamma) * p[c] + gamma * q[c];
            }
            p[0] = 0;
            codes[i] = TopK(p, _options.TopK, random) - 1;
            token = codes[i] + 1;
        }
        return codes;
    }

    // Top-k random sampling (Fan et al. 2018): the k most probable tokens, renormalized, one draw.
    private static int TopK(double[] probabilities, int k, Random random)
    {
        var top = Enumerable.Range(0, probabilities.Length).OrderByDescending(i => probabilities[i]).ThenBy(i => i).Take(k).ToArray();
        double u = random.NextDouble() * top.Sum(i => probabilities[i]);
        foreach (int i in top)
        {
            u -= probabilities[i];
            if (u < 0) return i;
        }
        return top[top.Length - 1];
    }

    internal Tensor<T> Content(int[] phones)
    {
        var m = new int[1, phones.Length];
        for (int i = 0; i < phones.Length; i++) m[0, i] = phones[i];
        return Network.Content.Forward(m);
    }

    private Tensor<T> Rows(Tensor<T> content) => Engine.Reshape(content, new[] { content.Shape[1], content.Shape[2] });

    /// <summary>The 1-based frame-to-phoneme map of durations over <paramref name="frames"/> frames: frames past the
    /// durations read no phoneme.</summary>
    internal static int[] FrameMap(int[] durations, int frames)
    {
        if (durations.Any(d => d < 0)) throw new ArgumentException("Durations are not negative.");
        var map = MegaTts3Aligner<T>.LengthRegulate(durations);
        if (map.Length > frames) throw new ArgumentException($"The durations cover {map.Length} frames; the recording has {frames}.");
        Array.Resize(ref map, frames);
        return map;
    }

    /// <summary>NATSpeech's log10 mel spectrogram <c>[frames, mels]</c> of a recording at <see cref="TtsModelOptions.SampleRate"/>.</summary>
    internal Tensor<T> MelOf(double[] audio)
    {
        var mel = _mel.LogMel(audio);
        int frames = mel.GetLength(0), mels = mel.GetLength(1);
        var values = new T[frames * mels];
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < mels; m++) values[f * mels + m] = NumOps.FromDouble(mel[f, m]);
        return new Tensor<T>(new[] { frames, mels }, new Vector<T>(values));
    }

    /// <summary>HiFi-GAN's waveform <c>[samples]</c> of a mel spectrogram <c>[frames, mels]</c>.</summary>
    internal Tensor<T> Wave(Tensor<T> mel)
    {
        int frames = mel.Shape[0];
        if (frames == 0) return new Tensor<T>(new[] { 0 });
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, _options.MelChannels, frames }));
        return Engine.Reshape(wave, new[] { wave.Length });
    }

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => MathHelper.GetNumericOperations<T>().ToDouble(v)).ToArray();

    // ---------------------------------------------------------------- weights

    /// <summary>Loads a HiFi-GAN V1 generator checkpoint in the reference's format (jik876/hifi-gan's
    /// <c>{"generator": state_dict}</c>, or the bare state dict), trained on this model's mel spectrograms.</summary>
    public void LoadVocoderWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        string prefix = state.Keys.Any(k => k.StartsWith("generator.", StringComparison.Ordinal)) ? "generator." : "";
        Vocoder.LoadTorch(name => state.TryGetValue(name, out var v) ? v : null, prefix);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MegaTTS2-Native" : "MegaTTS2-ONNX",
            Description = "Mega-TTS 2: zero-shot speech with multi-sentence prompts, a prosody latent language model and prosody interpolation (Jiang et al., 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "MegaTTS2";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Stage"] = _options.Stage.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MegaTTS2<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
