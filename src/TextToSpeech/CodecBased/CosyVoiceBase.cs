using AiDotNet.Audio.Features;
using AiDotNet.Audio.Speaker;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which CosyVoice network a training step trains (the recipes train <c>llm</c>, <c>flow</c> and
/// <c>hifigan</c> in turn).</summary>
public enum CosyVoiceTrainingStage
{
    /// <summary>The text-to-speech-token language model: next-token cross-entropy.</summary>
    LanguageModel,

    /// <summary>The speech-token-to-mel flow: conditional flow matching.</summary>
    Flow,

    /// <summary>HiFT against its discriminators (<c>HiFiGan</c>): per utterance a discriminator step, then a generator
    /// step. Needs <see cref="CosyVoiceFamilyOptions.IncludeDiscriminators"/>.</summary>
    Vocoder,
}

/// <summary>
/// What the CosyVoice models share (FunAudioLLM/CosyVoice, Apache-2.0): a language model speaking S3 tokens, a flow
/// sampling an 80-band mel from them by conditional flow matching with classifier-free guidance, HiFT turning the mel into
/// audio, and the frozen S3 tokenizer and CAM++ reading a prompt's tokens and x-vector; the front end
/// (<see cref="CosyVoiceTextFrontEnd"/>) and voices (zero-shot, cross-lingual, a speaker embedding); and training in three
/// stages. Each model supplies its text tokenizer and language model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class CosyVoiceBase<T> : TtsModelBase<T>, ICodecTts<T>
{
    // S3 and CAM++ read 16 kHz audio (load_wav(prompt_wav, 16000)).
    private const int PromptSampleRate = 16000;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<CosyVoiceTrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _stageOptimizers = new();
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private readonly Random _trainingRandom;
    private readonly bool _useNativeMode;
    private bool _disposed;
    // Which side of the GAN a vocoder step updates; read by the parameter selection.
    private bool _discriminatorTurn;

    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _flowLayers = new();
    private readonly List<LayerBase<T>> _hiftLayers = new();
    private readonly List<LayerBase<T>> _speechTokenizerLayers = new();
    private readonly List<LayerBase<T>> _speakerEncoderLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private bool _hasPaperLayers;
    private CosyVoiceFlow<T>? _flow;
    private HiftGenerator<T>? _hift;
    private S3Tokenizer<T>? _speechTokenizer;
    private CamPlusPlusEncoder<T>? _speakerEncoder;
    private CosyVoiceHiFTDiscriminators<T>? _discriminators;
    private CosyVoiceHiFTLosses<T>? _vocoderLosses;
    private KaldiFbank<T>? _fbank;
    private DifferentiableMel<T>? _mel;

    /// <summary>A native (trainable) model; the leaf calls <see cref="Initialize"/> once its own state is set.</summary>
    protected CosyVoiceBase(NeuralNetworkArchitecture<T> architecture, int samplingSeed,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
        : base(architecture)
    {
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(samplingSeed);
    }

    /// <summary>An ONNX-backed model running <paramref name="modelPath"/>.</summary>
    protected CosyVoiceBase(NeuralNetworkArchitecture<T> architecture, int samplingSeed, string modelPath, OnnxModelOptions onnxOptions)
        : base(architecture)
    {
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(samplingSeed);
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);
    }

    // ---------------------------------------------------------------- what each model supplies

    /// <summary>The model's options.</summary>
    protected abstract CosyVoiceFamilyOptions Settings { get; }

    /// <summary>The number of text ids the language model reads.</summary>
    protected abstract int TextVocabularySize { get; }

    /// <summary>The text token ids of <paramref name="text"/>, special tokens written in it allowed.</summary>
    public abstract int[] Tokenize(string text);

    /// <summary>The text of token ids.</summary>
    public abstract string Detokenize(IEnumerable<int> ids);

    /// <summary>The S3 tokenizer the release ships.</summary>
    private protected abstract S3TokenizerConfiguration SpeechTokenizerConfiguration { get; }

    /// <summary>The flow's configuration (CosyVoice 1's or CosyVoice 2's causal flow).</summary>
    private protected abstract CosyVoiceFlowConfiguration FlowConfiguration { get; }

    /// <summary>Builds the language model's layers into <paramref name="layers"/>.</summary>
    protected abstract void BuildLanguageModel(List<LayerBase<T>> layers, Random initialization);

    /// <summary>Speech tokens for text after an optional prompt (the language model's <c>inference</c>).</summary>
    protected abstract int[] GenerateWithLanguageModel(IReadOnlyList<int> text, IReadOnlyList<int> promptText,
        IReadOnlyList<int> promptSpeech, Tensor<T>? xvector, Random random);

    /// <summary>The language model's training loss on one utterance.</summary>
    protected abstract Tensor<T> LanguageModelLoss(IReadOnlyList<int> text, IReadOnlyList<int> speech, Tensor<T> xvector,
        bool training, Random random);

    /// <summary>Loads the language model from <c>llm.pt</c>'s state.</summary>
    protected abstract void LoadLanguageModel(Func<string, int[], double[]> read);

    /// <summary>Builds the networks and sets the base's audio settings; the leaf constructor calls it last.</summary>
    protected void Initialize()
    {
        var o = Settings;
        if (o.NumCodebooks != 1)
            throw new ArgumentException("CosyVoice speaks one stream of S3 tokens; NumCodebooks must be 1.", nameof(o.NumCodebooks));
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.LLMDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => Settings.SampleRate;
    public int MaxTextLength => Settings.MaxTextLength;
    public int NumCodebooks => Settings.NumCodebooks;
    public int CodebookSize => Settings.CodebookSize;
    public int CodecFrameRate => Settings.CodecFrameRate;

    /// <summary>The network a training step trains; the recipe trains the language model, the flow, then the vocoder.</summary>
    public CosyVoiceTrainingStage CurrentStage { get; set; } = CosyVoiceTrainingStage.LanguageModel;

    /// <inheritdoc />
    /// <remarks>The S3 tokens of the utterance and the speaker: its x-vector, or a recording CAM++ embeds.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens | TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => 1;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => Settings.CodebookSize;

    /// <summary>Whether the release's networks were built (not caller-supplied layers or an ONNX session).</summary>
    protected bool HasPaperLayers => _hasPaperLayers;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(TextVocabularySize) : base.GetInputDomain(inputShape);

    private protected CosyVoiceFlow<T> Flow => _flow ?? throw new InvalidOperationException("The flow exists in native mode only.");
    private HiftGenerator<T> Hift => _hift ?? throw new InvalidOperationException("The vocoder exists in native mode only.");
    private S3Tokenizer<T> SpeechTokenizer => _speechTokenizer ?? throw new InvalidOperationException("The speech tokenizer exists in native mode only.");
    private CamPlusPlusEncoder<T> SpeakerEncoder => _speakerEncoder ?? throw new InvalidOperationException("The speaker encoder exists in native mode only.");

    // The networks, for tests.
    internal CosyVoiceFlow<T>? FlowModel => _flow;
    internal HiftGenerator<T>? Vocoder => _hift;
    internal IReadOnlyList<LayerBase<T>> LanguageModelLayers => _lmLayers;
    internal IReadOnlyList<LayerBase<T>> FlowLayers => _flowLayers;
    internal IReadOnlyList<LayerBase<T>> FrozenLayers => _hiftLayers.Concat(_speechTokenizerLayers).Concat(_speakerEncoderLayers).ToList();
    internal IReadOnlyList<LayerBase<T>> VocoderLayers => _hiftLayers;
    internal IReadOnlyList<LayerBase<T>> DiscriminatorLayers => _discriminatorLayers;

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

        var o = Settings;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        BuildLanguageModel(_lmLayers, random);
        _flow = new CosyVoiceFlow<T>(Engine, _flowLayers, FlowConfiguration, random);

        // HiFTGenerator with ConvRNNF0Predictor: the release's upsampling and an iSTFT of 16 with hop 4.
        var stages = o.HiftUpsampleRates.Length;
        if (o.HiftUpsampleKernels.Length != stages || o.HiftSourceKernels.Length != stages)
            throw new ArgumentException("HiFT needs one upsampling kernel and one source kernel per upsampling rate.");
        var hift = new HiftConfiguration(InChannels: o.MelChannels, BaseChannels: o.HiftBaseChannels, Harmonics: o.HiftHarmonics,
            SampleRate: o.SampleRate, NsfAlpha: 0.1, NsfSigma: 0.003, VoicedThreshold: 10, UpsampleRates: (int[])o.HiftUpsampleRates.Clone(),
            UpsampleKernels: (int[])o.HiftUpsampleKernels.Clone(), FftSize: 16, HopSize: 4, ResblockKernels: new[] { 3, 7, 11 },
            ResblockDilations: new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } },
            SourceKernels: (int[])o.HiftSourceKernels.Clone(), SourceDilations: Enumerable.Range(0, stages).Select(_ => new[] { 1, 3, 5 }).ToArray(),
            LeakySlope: 0.1, AudioLimit: 0.99, F0Channels: o.HiftBaseChannels);
        if (hift.UpsampleScale != o.HopSize)
            throw new ArgumentException($"HiFT gives {hift.UpsampleScale} samples per mel frame; HopSize must be {hift.UpsampleScale}.");
        _hift = new HiftGenerator<T>(Engine, _hiftLayers, hift, random);

        // The frozen front end: the S3 tokenizer and the CAM++ x-vector extractor.
        _speechTokenizer = new S3Tokenizer<T>(Engine, _speechTokenizerLayers, SpeechTokenizerConfiguration, random);
        _speakerEncoder = new CamPlusPlusEncoder<T>(Engine, _speakerEncoderLayers, new CamPlusPlusConfiguration(
            EmbeddingSize: o.SpeakerEmbeddingDim, GrowthRate: o.SpeakerEncoderGrowthRate, InitialChannels: o.SpeakerEncoderChannels),
            random);
        _speakerEncoder.SetTrainingMode(false);

        if (o.IncludeDiscriminators)
        {
            if (o.VocoderSegmentLength <= 0 || o.VocoderSegmentLength % o.HopSize != 0)
                throw new ArgumentException($"VocoderSegmentLength must be a positive multiple of HopSize ({o.HopSize}).");
            _discriminators = new CosyVoiceHiFTDiscriminators<T>(Engine);
            _discriminatorLayers.AddRange(_discriminators.Layers);
            // The loss mel is the recipe's mel_spec_transform1: 80 bands to Nyquist at the model's FFT and hop.
            _vocoderLosses = new CosyVoiceHiFTLosses<T>(Engine, o.SampleRate, melFft: o.FftSize, melHop: o.HopSize, melWindow: o.FftSize);
        }

        _hasPaperLayers = true;
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_flowLayers);
        ComponentLayers.AddRange(_hiftLayers);
        ComponentLayers.AddRange(_speechTokenizerLayers);
        ComponentLayers.AddRange(_speakerEncoderLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
    }

    /// <inheritdoc />
    /// <remarks>The frozen S3 tokenizer and CAM++ stay in evaluation mode.</remarks>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _speakerEncoder?.SetTrainingMode(false);
        foreach (var layer in _speakerEncoderLayers) layer.SetTrainingMode(false);
    }

    // ---------------------------------------------------------------- front end

    /// <summary>The sentences the model synthesizes <paramref name="text"/> as (<c>text_normalize(text, split=True)</c>
    /// without an external normalizer): see <see cref="CosyVoiceTextFrontEnd"/>.</summary>
    public IReadOnlyList<string> SplitText(string text) => CosyVoiceTextFrontEnd.Sentences(text, s => Tokenize(s).Length);

    /// <inheritdoc />
    /// <remarks>As <c>inference_zero_shot</c> and its siblings do: each sentence of <see cref="SplitText"/> is
    /// synthesized in <see cref="TtsModelBase{T}.Voice"/> and the waveforms are concatenated. Text of punctuation alone
    /// gives no audio.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (text.Trim().Length == 0) throw new ArgumentException("There is no text to synthesize.", nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        var pieces = SplitText(text).Select(sentence => PostprocessAudio(Predict(PreprocessText(sentence)))).ToArray();
        return pieces.Length switch
        {
            0 => new Tensor<T>(new[] { 0 }),
            1 => pieces[0],
            _ => Engine.TensorConcatenate(pieces, 0),
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        var ids = Tokenize(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return Ids(ids);
    }

    /// <summary>
    /// The S3 speech tokens of a recording <c>[samples]</c> at <paramref name="sampleRate"/> (<c>_extract_speech_token</c>):
    /// resampled to 16 kHz, Whisper's 128-band log-mel, the release's S3 tokenizer.
    /// </summary>
    public int[] SpeechTokens(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        var audio = Resample(recording, sampleRate, PromptSampleRate);
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = NumOps.ToDouble(audio[i]);
        var logMel = S3Tokenizer<T>.LogMel(samples);
        int bands = logMel.GetLength(0), frames = logMel.GetLength(1);
        var mel = new Tensor<T>(new[] { bands, frames });
        for (int m = 0; m < bands; m++)
            for (int f = 0; f < frames; f++) mel[m, f] = NumOps.FromDouble(logMel[m, f]);
        return SpeechTokenizer.Tokenize(mel);
    }

    /// <summary>
    /// The CAM++ x-vector <c>[SpeakerEmbeddingDim]</c> of a recording (<c>_extract_spk_embedding</c>): resampled to
    /// 16 kHz, 80-band Kaldi filterbank energies minus their utterance mean, CAM++ in evaluation mode.
    /// </summary>
    public Tensor<T> SpeakerEmbedding(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        var audio = Resample(recording, sampleRate, PromptSampleRate);
        _fbank ??= new KaldiFbank<T>(Engine, PromptSampleRate, 80, WindowType.Povey);
        var features = _fbank.Forward(audio);
        features = Engine.TensorSubtract(features, Engine.TensorBroadcastTo(Engine.ReduceMean(features, new[] { 0 }, true), features.Shape.ToArray()));
        var embedding = SpeakerEncoder.Forward(Engine.Reshape(features, new[] { 1, features.Shape[0], features.Shape[1] }));
        return Engine.Reshape(embedding, new[] { Settings.SpeakerEmbeddingDim });
    }

    /// <summary>
    /// The flow's mel <c>[frames, MelChannels]</c> of a recording at the model's sample rate (<c>_extract_speech_feat</c>:
    /// Matcha's <c>mel_spectrogram</c> with the model's FFT, hop and window, 0–8 kHz, not centred).
    /// </summary>
    public Tensor<T> MelSpectrogram(Tensor<T> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var o = Settings;
        _mel ??= new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.FftSize, o.MelChannels, 0, 8000);
        return _mel.Forward(Engine.Reshape(audio, new[] { audio.Length }));
    }

    /// <summary>
    /// A voice for <see cref="TtsModelBase{T}.Voice"/> from a recording of the speaker <c>[samples]</c> at
    /// <paramref name="sampleRate"/>: its S3 tokens, its mel (resampled to the model's rate) and its CAM++ x-vector. With
    /// <paramref name="transcript"/> the voice is zero-shot (<c>frontend_zero_shot</c>): the language model continues the
    /// prompt's text and tokens. Without it, cross-lingual (<c>frontend_cross_lingual</c>): only the flow reads the prompt.
    /// A causal flow's prompt keeps <c>TokenMelRatio</c> mel frames per token, both trimmed to the shorter.
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string? transcript = null)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var tokens = SpeechTokens(recording, sampleRate);
        Tensor<T> mel;
        using (new NoGradScope<T>()) mel = MelSpectrogram(Resample(recording, sampleRate, Settings.SampleRate));
        var flow = FlowConfiguration;
        if (flow.Causal)
        {
            // frontend_zero_shot at 24 kHz: token_len = min(feat_len / ratio, token_len), the feat trimmed to match.
            int count = Math.Min(mel.Shape[0] / flow.TokenMelRatio, tokens.Length);
            tokens = tokens.Take(count).ToArray();
            mel = Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { count * flow.TokenMelRatio, mel.Shape[1] });
        }
        return new TtsVoice<T>
        {
            Reference = mel,
            ReferenceCodecTokens = Ids(tokens),
            // inference_zero_shot normalizes the prompt's transcript unsplit.
            ReferenceTokens = transcript is null ? null : Ids(Tokenize(CosyVoiceTextFrontEnd.Normalize(transcript))),
            SpeakerEmbedding = SpeakerEmbedding(recording, sampleRate),
        };
    }

    /// <summary>A voice of a speaker embedding alone (<c>frontend_sft</c>: a speaker of <c>spk2info</c>).</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> speakerEmbedding)
    {
        if (speakerEmbedding is null) throw new ArgumentNullException(nameof(speakerEmbedding));
        if (speakerEmbedding.Length != Settings.SpeakerEmbeddingDim)
            throw new ArgumentException($"A speaker embedding has {Settings.SpeakerEmbeddingDim} values.", nameof(speakerEmbedding));
        return new TtsVoice<T> { SpeakerEmbedding = speakerEmbedding };
    }

    private Tensor<T> Resample(Tensor<T> audio, int from, int to)
    {
        var flat = Engine.Reshape(audio, new[] { audio.Length });
        return from == to ? flat : AudioHelper<T>.ResampleBandLimited(Engine, flat, from, to);
    }

    /// <summary>Ids as a tensor <c>[count]</c>.</summary>
    protected Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    /// <summary>The ids a tensor holds.</summary>
    protected int[] IdsOf(Tensor<T> tensor)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++) ids[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
        return ids;
    }

    // ---------------------------------------------------------------- synthesis

    /// <summary>
    /// Generates speech tokens for text ids under <see cref="TtsModelBase{T}.Voice"/> (<c>llm.inference</c>): prompted by
    /// the voice's transcript and tokens in zero-shot, by its x-vector alone otherwise.
    /// </summary>
    public int[] GenerateSpeechTokens(IReadOnlyList<int> text, Random random)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var voice = Voice;
        // Zero-shot only when the voice carries both the transcript and its speech tokens.
        int[] promptText = Array.Empty<int>(), promptSpeech = Array.Empty<int>();
        if (voice is { ReferenceTokens: { } transcript, ReferenceCodecTokens: { } prompt })
        {
            promptText = IdsOf(transcript);
            promptSpeech = IdsOf(prompt);
        }
        return GenerateWithLanguageModel(text, promptText, promptSpeech, voice?.SpeakerEmbedding, random);
    }

    /// <summary>
    /// The mel <c>[1, MelChannels, frames]</c> of speech tokens (the flow's <c>inference</c>): the voice's prompt tokens
    /// and these encoded together, the prompt's mel as the condition, Euler sampling from Gaussian noise, and the prompt's
    /// frames removed.
    /// </summary>
    public Tensor<T> TokensToMel(IReadOnlyList<int> tokens, Random random)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Count == 0) throw new ArgumentException("There are no speech tokens to decode.", nameof(tokens));
        var flow = Flow;
        var voice = Voice;
        var promptTokens = voice?.ReferenceCodecTokens is { } codes ? IdsOf(codes) : Array.Empty<int>();
        var promptMel = promptTokens.Length > 0
            ? voice?.Reference ?? throw new InvalidOperationException("A voice with prompt tokens needs the prompt's mel (Reference) for the flow.")
            : null;
        int frames = (promptMel?.Shape[0] ?? 0) + flow.MelFrames(tokens.Count);
        // The flow's Gaussian start (ConditionalCFM: randn_like(mu); CausalConditionalCFM: a fixed draw, which a seeded
        // generator stands in for).
        var noise = new Tensor<T>(new[] { 1, Settings.MelChannels, frames });
        for (int i = 0; i < noise.Length; i++)
            noise[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble()));
        var xvector = voice?.SpeakerEmbedding ?? new Tensor<T>(new[] { Settings.SpeakerEmbeddingDim });
        return flow.Inference(tokens, promptTokens, promptMel, xvector, noise);
    }

    /// <summary>The waveform <c>[samples]</c> of a mel <c>[1, MelChannels, frames]</c> (<c>HiFTGenerator.inference</c>).</summary>
    public Tensor<T> MelToAudio(Tensor<T> mel, Random random)
    {
        if (mel is null) throw new ArgumentNullException(nameof(mel));
        var audio = Hift.Forward(mel, random);
        return Engine.Reshape(audio, new[] { audio.Length });
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is text token ids (<see cref="PreprocessText"/>); the waveform <c>[samples]</c>
    /// comes from the language model's tokens, the flow's mel and HiFT, each sampled from a generator seeded by
    /// <see cref="CosyVoiceFamilyOptions.SamplingSeed"/>.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Settings.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var text = IdsOf(input);
            if (text.Length == 0) throw new ArgumentException("There is no text to synthesize.", nameof(input));
            if (text.Any(id => id < 0 || id >= TextVocabularySize))
                throw new ArgumentOutOfRangeException(nameof(input), $"Text ids lie in [0, {TextVocabularySize}).");
            var tokens = GenerateSpeechTokens(text, random);
            if (tokens.Length == 0) return new Tensor<T>(new[] { 0 });
            return MelToAudio(TokensToMel(tokens, random), random);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <summary>Encodes audio at the model's sample rate into S3 speech tokens <c>[frames]</c>.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(audio);
        return Ids(SpeechTokens(audio, Settings.SampleRate));
    }

    /// <summary>Decodes S3 speech tokens into audio through the flow and HiFT, in <see cref="TtsModelBase{T}.Voice"/>.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Settings.SamplingSeed);
        using var _ = new NoGradScope<T>();
        return MelToAudio(TokensToMel(IdsOf(tokens), random), random);
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfTokenMelTrainingUnsupported();
        return false;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (CurrentStage == CosyVoiceTrainingStage.Vocoder) return TrainVocoder(sample);
        var objective = BuildObjective(sample, _trainingRandom, training: true);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => objective(), StageOptimizer());
    }

    /// <summary>
    /// One GAN step on an utterance (<c>HiFiGan.forward_discriminator</c>, then <c>forward_generator</c>): the recording
    /// cropped or padded to <see cref="CosyVoiceFamilyOptions.VocoderSegmentLength"/>, its mel, and its F0 (Harvest, as
    /// <c>compute_f0</c> extracts it). The discriminators learn from HiFT's output computed without gradients; HiFT then
    /// learns from the adversarial, feature-matching, mel, TPR and F0 losses.
    /// </summary>
    /// <returns>The generator's loss.</returns>
    private T TrainVocoder(TtsTrainingSample<T> sample)
    {
        var discriminators = _discriminators ?? throw new InvalidOperationException(
            "Training the vocoder needs HiFT's discriminators; build the model with IncludeDiscriminators = true.");
        var losses = _vocoderLosses ?? throw new InvalidOperationException("The vocoder losses are built with the discriminators.");
        var (real, mel, pitch) = VocoderTargets(sample, _trainingRandom);
        var hift = Hift;
        var random = _trainingRandom;

        _discriminatorTurn = true;
        try
        {
            TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
            {
                Tensor<T> generated;
                using (new NoGradScope<T>())
                {
                    var audio = hift.Forward(mel, random);
                    generated = Engine.Reshape(audio, new[] { audio.Length });
                }
                return losses.Discriminator(discriminators.Forward(real), discriminators.Forward(generated));
            }, VocoderOptimizer("discriminator"));
        }
        finally
        {
            _discriminatorTurn = false;
        }

        var generator = GeneratorObjective(real, mel, pitch, random);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => generator(), VocoderOptimizer("vocoder"));
    }

    // forward_generator's loss: HiFT's output from the mel and its own F0's excitation, against the recording.
    private Func<Tensor<T>> GeneratorObjective(Tensor<T> real, Tensor<T> mel, Tensor<T> pitch, Random random)
    {
        var discriminators = _discriminators ?? throw new InvalidOperationException(
            "Training the vocoder needs HiFT's discriminators; build the model with IncludeDiscriminators = true.");
        var losses = _vocoderLosses ?? throw new InvalidOperationException("The vocoder losses are built with the discriminators.");
        var hift = Hift;
        return () =>
        {
            var f0 = hift.PredictF0(mel);
            var audio = hift.Decode(mel, hift.Source(f0, random));
            var generated = Engine.Reshape(audio, new[] { audio.Length });
            return losses.Generator(real, generated, discriminators.Forward(real), discriminators.Forward(generated), f0, pitch).Total;
        };
    }

    /// <summary>
    /// A vocoder step's targets (the recipe's <c>truncate</c>, <c>compute_fbank</c>, <c>compute_f0</c>): the recording
    /// <c>[segment]</c>, its mel <c>[1, MelChannels, frames]</c> and its F0 <c>[frames]</c> — Harvest (DIO when Harvest
    /// finds fewer than five voiced frames) refined by StoneMask at the mel's frame period, linearly resampled to the
    /// mel's frames — or <see cref="TtsTrainingSample{T}.Pitch"/> when given.
    /// </summary>
    internal (Tensor<T> Audio, Tensor<T> Mel, Tensor<T> Pitch) VocoderTargets(TtsTrainingSample<T> sample, Random random)
    {
        var o = Settings;
        var recording = sample.Audio ?? throw new ArgumentException("The vocoder trains on the recording; set Audio.", nameof(sample));
        int segment = o.VocoderSegmentLength, length = recording.Length;
        var audio = new Tensor<T>(new[] { segment });
        int start = length > segment ? random.Next(length - segment + 1) : 0;
        for (int i = 0; i < Math.Min(segment, length); i++) audio[i] = recording[start + i];

        Tensor<T> features;
        using (new NoGradScope<T>()) features = MelSpectrogram(audio);
        int frames = features.Shape[0];
        var mel = Engine.Reshape(Engine.TensorTranspose(features), new[] { 1, o.MelChannels, frames });

        double[] f0;
        if (sample.Pitch is { } given)
        {
            if (given.Length != frames) throw new ArgumentException($"Pitch needs one value per mel frame ({frames}).", nameof(sample));
            f0 = given;
        }
        else
        {
            f0 = Resize(HarvestF0(audio), frames);
        }
        var pitch = new Tensor<T>(new[] { frames });
        for (int i = 0; i < frames; i++) pitch[i] = NumOps.FromDouble(f0[i]);
        return (audio, mel, pitch);
    }

    // compute_f0: pw.harvest at the mel's frame period, pw.dio if harvest finds fewer than five voiced frames, then
    // pw.stonemask.
    private double[] HarvestF0(Tensor<T> audio)
    {
        var o = Settings;
        var x = new double[audio.Length];
        for (int i = 0; i < x.Length; i++) x[i] = NumOps.ToDouble(audio[i]);
        double period = o.HopSize * 1000.0 / o.SampleRate;
        var detector = new AiDotNet.Audio.Pitch.WorldPitchDetector<T>(o.SampleRate);
        var (f0, positions) = detector.Harvest(x, period);
        if (f0.Count(v => v != 0) < 5)
            (f0, positions) = new AiDotNet.Audio.Pitch.WorldPitchDetector<T>(o.SampleRate, refineWithStoneMask: false).EstimateF0(x, period);
        return detector.StoneMask(x, f0, positions);
    }

    // F.interpolate(mode='linear', align_corners=False) of a contour to `size` points.
    internal static double[] Resize(double[] values, int size)
    {
        var result = new double[size];
        int length = values.Length;
        double scale = (double)length / size;
        for (int i = 0; i < size; i++)
        {
            double source = Math.Max(0.0, scale * (i + 0.5) - 0.5);
            int low = Math.Min((int)source, length - 1);
            int high = low < length - 1 ? low + 1 : low;
            double weight = source - low;
            result[i] = (1 - weight) * values[low] + weight * values[high];
        }
        return result;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? VocoderOptimizer(string component)
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Create() => PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, component)
            ?? throw new InvalidOperationException($"{GetType().Name} declares an optimizer for the {component}.");
        if (component == "discriminator") return _discriminatorOptimizer ??= Create();
        if (!_stageOptimizers.TryGetValue(CosyVoiceTrainingStage.Vocoder, out var optimizer))
            _stageOptimizers[CosyVoiceTrainingStage.Vocoder] = optimizer = Create();
        return optimizer;
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var objective = BuildObjective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Settings.SamplingSeed), training: false);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The stage's objective on one utterance: the language model's loss; the flow's conditional flow matching on the
    /// utterance's mel with a random prompt prefix and condition dropout; or HiFT's generator loss. Speech tokens come
    /// from <see cref="TtsTrainingSample{T}.CodecTokens"/> or the recording, the x-vector from
    /// <see cref="TtsTrainingSample{T}.SpeakerReference"/> or the recording, the mel from
    /// <see cref="TtsTrainingSample{T}.Mel"/> (<c>[frames, MelChannels]</c>) or the recording. A causal flow's tokens and
    /// mel are trimmed to the shorter at <c>TokenMelRatio</c> frames per token.
    /// </summary>
    internal Func<Tensor<T>> BuildObjective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The objective needs the release's layers; this model was built from caller-supplied layers.");
        var o = Settings;
        if (CurrentStage == CosyVoiceTrainingStage.Vocoder)
        {
            var (real, logMel, pitch) = VocoderTargets(sample, random);
            return GeneratorObjective(real, logMel, pitch, random);
        }
        var text = IdsOf(sample.Tokens);
        if (text.Any(id => id < 0 || id >= TextVocabularySize))
            throw new ArgumentOutOfRangeException(nameof(sample), $"Text ids lie in [0, {TextVocabularySize}).");
        var speech = SpeechTokensOf(sample);
        var xvector = XVectorOf(sample);

        if (CurrentStage == CosyVoiceTrainingStage.LanguageModel)
            return () => LanguageModelLoss(text, speech, xvector, training, random);

        Tensor<T> mel;
        if (sample.Mel is { } given)
        {
            if (given.Rank != 2 || given.Shape[1] != o.MelChannels)
                throw new ArgumentException($"Expected a mel [frames, {o.MelChannels}].", nameof(sample));
            mel = given;
        }
        else
        {
            var recording = sample.Audio ?? throw new ArgumentException("The flow trains on the utterance's mel; set Mel or Audio.", nameof(sample));
            using (new NoGradScope<T>()) mel = MelSpectrogram(recording);
        }
        var flowConfiguration = FlowConfiguration;
        if (flowConfiguration.Causal)
        {
            int count = Math.Min(mel.Shape[0] / flowConfiguration.TokenMelRatio, speech.Length);
            if (count == 0) throw new ArgumentException("The utterance is too short for one speech token.", nameof(sample));
            speech = speech.Take(count).ToArray();
            mel = Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { count * flowConfiguration.TokenMelRatio, o.MelChannels });
        }
        var target = Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, o.MelChannels, mel.Shape[0] });
        return () => Flow.TrainingLoss(speech, target, xvector, training, random);
    }

    // The utterance's x-vector: SpeakerReference when it is one ([SpeakerEmbeddingDim]), else CAM++'s of the recording
    // SpeakerReference or Audio holds, at the model's sample rate, as the recipe's data preparation extracts it.
    private Tensor<T> XVectorOf(TtsTrainingSample<T> sample)
    {
        if (sample.SpeakerReference is { Rank: 1 } reference && reference.Length == Settings.SpeakerEmbeddingDim)
            return reference;
        var recording = sample.SpeakerReference ?? sample.Audio
            ?? throw new ArgumentException("Training needs the speaker's x-vector; set SpeakerReference (an x-vector or a recording) or Audio.", nameof(sample));
        return SpeakerEmbedding(recording, Settings.SampleRate);
    }

    private int[] SpeechTokensOf(TtsTrainingSample<T> sample)
    {
        if (sample.CodecTokens is { } tokens)
        {
            var ids = IdsOf(tokens);
            if (ids.Any(id => id < 0 || id >= Settings.CodebookSize))
                throw new ArgumentOutOfRangeException(nameof(sample), $"S3 tokens lie in [0, {Settings.CodebookSize}).");
            return ids;
        }
        var audio = sample.Audio ?? throw new ArgumentException("Training needs S3 speech tokens, or the recording.", nameof(sample));
        return SpeechTokens(audio, Settings.SampleRate);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? StageOptimizer()
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        if (!_stageOptimizers.TryGetValue(CurrentStage, out var optimizer))
        {
            optimizer = PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this,
                    CurrentStage == CosyVoiceTrainingStage.LanguageModel ? "language model" : "flow", Settings.WarmupSteps)
                ?? throw new InvalidOperationException($"{GetType().Name} declares an optimizer for each stage.");
            _stageOptimizers[CurrentStage] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>A stage updates its own network only (the vocoder stage HiFT, or its discriminators on their turn); the S3
    /// tokenizer and CAM++ stay as loaded.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = CurrentStage switch
        {
            CosyVoiceTrainingStage.LanguageModel => _lmLayers,
            CosyVoiceTrainingStage.Flow => _flowLayers,
            _ => _discriminatorTurn ? _discriminatorLayers : _hiftLayers,
        };
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>
    /// Loads the released networks from a model directory, each a state dict converted to safetensors:
    /// <c>llm.safetensors</c>, <c>flow.safetensors</c> and <c>hift.safetensors</c> (from <c>llm.pt</c>, <c>flow.pt</c>,
    /// <c>hift.pt</c>), the S3 tokenizer (xingchensong/S3Tokenizer's PyTorch form of the release's ONNX tokenizer) as
    /// <paramref name="speechTokenizerFile"/>, and <c>campplus.safetensors</c> (3D-Speaker's CAM++).
    /// </summary>
    protected void LoadReleasedWeights(string modelDirectory, string speechTokenizerFile)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(modelDirectory)) throw new ArgumentException("The model directory is required.", nameof(modelDirectory));
        LoadLanguageModel(Reader(Path.Combine(modelDirectory, "llm.safetensors")));
        Flow.LoadTorchWeights(Reader(Path.Combine(modelDirectory, "flow.safetensors")), "");
        Hift.LoadTorchWeights(Reader(Path.Combine(modelDirectory, "hift.safetensors")));
        SpeechTokenizer.LoadTorchWeights(Reader(Path.Combine(modelDirectory, speechTokenizerFile)));
        SpeakerEncoder.LoadTorchWeights(Reader(Path.Combine(modelDirectory, "campplus.safetensors")));
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
    }

    private static Func<string, int[], double[]> Reader(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"The checkpoint {path} is missing.", path);
        var file = AiDotNet.Agentic.Models.Local.SafetensorsReader.Read(File.ReadAllBytes(path));
        return (name, shape) =>
        {
            var entry = file.Get(name) ?? throw new InvalidDataException($"The checkpoint {path} has no tensor '{name}'.");
            // num_batches_tracked is an int64 scalar that nothing reads.
            if (entry.DataType == "I64") return new[] { (double)BitConverter.ToInt64(file.GetRawBytes(name), 0) };
            var values = file.ReadAsDouble(name);
            if (shape.Length > 0 && values.Length != shape.Aggregate(1, (a, b) => a * b))
                throw new InvalidDataException($"'{name}' has {values.Length} values in the checkpoint but [{string.Join(", ", shape)}] " +
                    "in this model; build the model with the checkpoint's configuration.");
            return values;
        };
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <summary>Whether this instance runs the native networks (not an ONNX session).</summary>
    protected bool IsNative => _useNativeMode;

    /// <summary>Throws once the model is disposed.</summary>
    protected void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? GetType().Name);
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
