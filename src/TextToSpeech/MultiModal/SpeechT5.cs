using AiDotNet.Attributes;
using AiDotNet.Audio.Speaker;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// SpeechT5 (Ao et al. 2022): one Transformer encoder-decoder shared by speech and text, pre-trained jointly on unlabeled
/// speech (HuBERT-style masked prediction plus reconstructing the log-mel spectrogram) and unlabeled text (BART-style
/// infilling), with a cross-modal codebook aligning the two, then fine-tuned to read text aloud in the voice of a speaker
/// x-vector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Text to speech runs the text encoder pre-net (a character embedding plus scaled positions), the shared encoder
/// (post-norm layers with a relative position bias), and the speech decoder: its pre-net (Tacotron 2's always-on dropout
/// pre-net, scaled positions and the normalized x-vector), the shared decoder, r = 2 frames and a stop probability per
/// step, and a five-layer convolutional post-net. HiFi-GAN (<c>microsoft/speecht5_hifigan</c>) turns the 16 kHz log-mel
/// spectrogram into audio. The x-vector comes from speechbrain's extractor (<c>spkrec-xvect-voxceleb</c>), as the
/// release's speaker embeddings do.</para>
/// <para>The paper is the default; the official code (<c>microsoft/SpeechT5</c>) fills what it leaves out, and the Hugging
/// Face port's stop rule is an option (<see cref="SpeechT5Options.StopCriterion"/>).</para>
/// <para><b>For Beginners:</b> Give SpeechT5 a few seconds of someone speaking (or their x-vector) and text, and it reads
/// the text in a voice like theirs.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new SpeechT5&lt;float&gt;(architecture, new SpeechT5Options());
/// model.LoadReferenceWeights(new SpeechT5Checkpoints { Model = "speecht5_tts/pytorch_model.bin",
///     Vocoder = "speecht5_hifigan/pytorch_model.bin", SpeakerEncoder = "spkrec-xvect-voxceleb/embedding_model.ckpt" });
/// model.Voice = model.CreateVoice(recording, 16000);
/// var audio = model.Synthesize("Hello world.");   // 16 kHz
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing",
    "https://arxiv.org/abs/2110.07205",
    Year = 2022,
    Authors = "Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei"
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4,
                Phase = TrainingPhase.PreTraining,
                Schedule = LearningRateSchedulerType.LinearWarmup, WarmupFraction = 0.08,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear, MinLearningRate = 0,
                Source = "Ao et al. 2022, Sec. 3.1 (pre-training): Adam warming up over the first 8% of updates to a peak of 2e-4, "
                        + "then linearly decayed, for 500k steps. Built by the model rather than by the factory because it constructs "
                        + "explicit options; the declaration verifies those values instead of replacing them.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 4e-4,
                Phase = TrainingPhase.FineTuning, WarmupSteps = 10000,
                Schedule = LearningRateSchedulerType.Noam,
                Source = "Ao et al. 2022, Appendix B.2 (TTS): updated for 120k steps with a learning rate of 0.0004, warmed up for "
                        + "the first 10k steps and decayed in an inverse square root manner for the rest.")]
public partial class SpeechT5<T> : TtsModelBase<T>
{
    private readonly SpeechT5Options _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private SpeechT5Core<T>? _core;
    private SpeechBrainXVector<T>? _speakerEncoder;
    private HiFiGanGenerator<T>? _vocoder;
    private double[]? _vocoderMean;
    private double[]? _vocoderScale;
    private CenteredLogMel<T>? _mel;
    private SentencePieceModel? _tokenizer;

    private readonly List<LayerBase<T>> _layers = new();
    private readonly List<LayerBase<T>> _pretrainLayers = new();
    private readonly List<LayerBase<T>> _speakerEncoderLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed SpeechT5 for inference.</summary>
    public SpeechT5(NeuralNetworkArchitecture<T> architecture, string modelPath, SpeechT5Options? options = null)
        : base(architecture)
    {
        _options = options ?? new SpeechT5Options();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable SpeechT5.</summary>
    public SpeechT5(NeuralNetworkArchitecture<T> architecture, SpeechT5Options? options = null)
        : base(architecture)
    {
        _options = options ?? new SpeechT5Options();
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
        if (o.HiddenDim % o.NumHeads != 0) throw new ArgumentException("HiddenDim must divide by NumHeads.");
        if (o.ReductionFactor < 1) throw new ArgumentException("ReductionFactor must be positive.");
        if (o.VocoderUpsampleRates.Aggregate(1, (a, b) => a * b) != o.HopSize)
            throw new ArgumentException($"The vocoder's upsampling rates must multiply to HopSize {o.HopSize}.");
        if ((o.VocoderInitialChannels >> o.VocoderUpsampleRates.Length) < 1)
            throw new ArgumentException("Each vocoder upsampling stage halves the channels: VocoderInitialChannels must keep at least one through them all.");
        if (o.GuidedAttentionHeads < 0 || o.GuidedAttentionHeads > o.NumHeads)
            throw new ArgumentException("GuidedAttentionHeads lies in [0, NumHeads].");
        if (o.CodebookProbability is < 0 or > 1) throw new ArgumentException("CodebookProbability lies in [0, 1].");
        if (o.MaskId < 0 || o.MaskId >= o.VocabSize) throw new ArgumentException("MaskId must be a token of the vocabulary.");
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>Synthesis reads the speaker's x-vector: the voice's <see cref="TtsVoice{T}.SpeakerEmbedding"/>, or the
    /// x-vector of its <see cref="TtsVoice{T}.ReferenceAudio"/> (see <see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool HasPaperLayers => _core is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal SpeechT5Core<T> Core => _core ?? throw NotNative();
    internal SpeechBrainXVector<T> SpeakerEncoder => _speakerEncoder ?? throw NotNative();
    internal HiFiGanGenerator<T> Vocoder => _vocoder ?? throw NotNative();

    internal static SpeechT5Configuration Configuration(SpeechT5Options o) => new()
    {
        Hidden = o.HiddenDim, FeedForward = o.FeedForwardDim, Heads = o.NumHeads, EncoderLayers = o.NumEncoderLayers, DecoderLayers = o.NumDecoderLayers,
        MaxRelativePosition = o.MaxRelativePosition, TextVocabulary = o.VocabSize, PadId = o.PadId, EosId = o.EosId, MaxTextPositions = o.MaxTextLength,
        MaxSpeechPositions = o.MaxMelLength, MelChannels = o.MelChannels, ReductionFactor = o.ReductionFactor, SpeakerDim = o.SpeakerDim,
        PrenetUnits = o.PrenetUnits, PrenetLayers = o.PrenetLayers, PrenetDropout = o.PrenetDropout, PostnetChannels = o.PostnetChannels,
        PostnetLayers = o.PostnetLayers, PostnetKernel = o.PostnetKernel, PostnetDropout = o.PostnetDropout, FeatureLayers = o.FeatureLayers,
        ConvPositionKernel = o.ConvPositionKernel, ConvPositionGroups = o.ConvPositionGroups, FinalDim = o.FinalDim, HubertClasses = o.HubertClasses,
        LatentVars = o.LatentVars, LatentGroups = o.LatentGroups, Dropout = o.DropoutRate, AttentionDropout = o.AttentionDropout,
        ActivationDropout = o.ActivationDropout, EncoderLayerDrop = o.EncoderLayerDrop, DecoderLayerDrop = o.DecoderLayerDrop,
        PositionalDropout = o.PositionalDropout,
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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.TrainingSeed);
        _core = new SpeechT5Core<T>(Engine, _layers, _pretrainLayers, Configuration(o), random);
        if (o.BertInitialization) _core.BertInitialize(_layers.Concat(_pretrainLayers), random);
        _speakerEncoder = new SpeechBrainXVector<T>(Engine, _speakerEncoderLayers,
            new SpeechBrainXVectorConfiguration { Channels = o.SpeakerEncoderChannels, EmbeddingSize = o.SpeakerDim }, random);
        foreach (var layer in _speakerEncoderLayers) layer.SetTrainingMode(false);
        _vocoder = new HiFiGanGenerator<T>(Engine, o.MelChannels, o.VocoderInitialChannels, o.VocoderUpsampleRates, o.VocoderUpsampleKernels,
            o.VocoderResBlockKernels, o.VocoderResBlockDilations, resblockType1: true, initialization: random);
        _vocoderMean = new double[o.MelChannels];
        _vocoderScale = Enumerable.Repeat(1.0, o.MelChannels).ToArray();
        AddUnsplitLayers(_layers.Concat(_pretrainLayers).Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_speakerEncoderLayers);
        ComponentLayers.AddRange(_vocoder.Layers);
    }

    // ---------------------------------------------------------------- text

    private static readonly Lazy<SentencePieceModel> CharacterModel = new(() =>
    {
        const string name = "AiDotNet.TextToSpeech.MultiModal.speecht5_spm_char.model.gz";
        using var stream = typeof(SpeechT5<T>).Assembly.GetManifestResourceStream(name)
            ?? throw new InvalidDataException($"The embedded resource {name} is missing.");
        return SentencePieceModel.FromGzip(stream);
    });

    /// <summary>The character SentencePiece model: <see cref="SpeechT5Options.SentencePieceModelPath"/>, or the release's
    /// <c>spm_char.model</c>.</summary>
    internal SentencePieceModel Tokenizer =>
        _tokenizer ??= _options.SentencePieceModelPath is { } path ? new SentencePieceModel(File.ReadAllBytes(path)) : CharacterModel.Value;

    /// <summary>The token ids of <paramref name="text"/>: its characters' SentencePiece ids, then the end token.</summary>
    public int[] TokenizeText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Tokenizer.Encode(text).Concat(new[] { _options.EosId }).ToArray();
    }

    /// <inheritdoc />
    /// <remarks>The text's token ids (<see cref="TokenizeText"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length > _options.MaxTextLength) throw new ArgumentException($"The text has more than {_options.MaxTextLength} tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    private int[] TokenIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"Text token ids lie in [0, {_options.VocabSize}).");
        }
        if (ids.Length > _options.MaxTextLength) throw new ArgumentException($"There are more than {_options.MaxTextLength} text tokens.", name);
        return ids;
    }

    // ---------------------------------------------------------------- the speaker

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    /// <summary>The speechbrain x-vector <c>[SpeakerDim]</c> of audio at <see cref="TtsModelOptions.SampleRate"/>, its
    /// pooling noise drawn from <paramref name="random"/>.</summary>
    internal Tensor<T> XVector(Tensor<T> audio, Random random)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _speakerEncoderLayers) layer.SetTrainingMode(false);
        return SpeakerEncoder.Embed(Doubles(audio), random);
    }

    /// <summary>
    /// A voice from a recording of the speaker at <paramref name="sampleRate"/>: the recording resampled to 16 kHz and its
    /// speechbrain x-vector (the speaker embeddings the release was trained with).
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (!HasPaperLayers) throw NotNative();
        var audio = sampleRate == _options.SampleRate ? recording
            : AudioHelper<T>.ResampleBandLimited(Engine, Engine.Reshape(recording, new[] { recording.Length }), sampleRate, _options.SampleRate);
        var embedding = XVector(audio, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
        return new TtsVoice<T> { ReferenceAudio = audio, SpeakerEmbedding = embedding };
    }

    private readonly Dictionary<Tensor<T>, Tensor<T>> _xvectors = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private Tensor<T> SpeakerOf(TtsVoice<T> voice)
    {
        if (voice.SpeakerEmbedding is { } embedding)
        {
            if (embedding.Length != _options.SpeakerDim) throw new InvalidOperationException($"The speaker embedding has {_options.SpeakerDim} values.");
            return Engine.Reshape(embedding, new[] { _options.SpeakerDim });
        }
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException(
            $"{GetType().Name} speaks in the voice of an x-vector; set SpeakerEmbedding or ReferenceAudio on the voice.");
        if (!_xvectors.TryGetValue(audio, out var xvector))
        {
            xvector = XVector(audio, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
            if (_xvectors.Count > 64) _xvectors.Clear();
            _xvectors[audio] = xvector;
        }
        return xvector;
    }

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

    /// <summary>The log-mel spectrogram <c>[frames, mel]</c> SpeechT5 generates for text token ids in a voice (before
    /// the vocoder).</summary>
    public Tensor<T> GenerateSpectrogram(IReadOnlyList<int> tokens, TtsVoice<T> voice)
    {
        ThrowIfDisposed();
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (voice is null) throw new ArgumentNullException(nameof(voice));
        if (!HasPaperLayers) throw NotNative();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var o = _options;
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
            return Core.Generate(tokens, SpeakerOf(voice), random, o.StopThreshold, o.MaxLengthRatio,
                o.StopCriterion == SpeechT5StopCriterion.SummedProbability).Mel;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>HiFi-GAN's waveform <c>[samples]</c> of a log-mel spectrogram <c>[frames, mel]</c>, normalized by the
    /// vocoder's mean and scale first (<c>normalize_before</c>).</summary>
    public Tensor<T> Vocode(Tensor<T> mel)
    {
        ThrowIfDisposed();
        if (mel is null) throw new ArgumentNullException(nameof(mel));
        if (!HasPaperLayers) throw NotNative();
        int frames = mel.Shape[0], m = _options.MelChannels;
        if (mel.Rank != 2 || mel.Shape[1] != m) throw new ArgumentException($"The spectrogram is [frames, {m}].", nameof(mel));
        using var _ = new NoGradScope<T>();
        var shift = new Tensor<T>(new[] { frames, m });
        var scale = new Tensor<T>(new[] { frames, m });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < m; c++)
            {
                shift[f, c] = Ops.FromDouble(_vocoderMean![c]);
                scale[f, c] = Ops.FromDouble(1.0 / _vocoderScale![c]);
            }
        var normalized = Engine.TensorMultiply(Engine.TensorSubtract(mel, shift), scale);
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(normalized), new[] { 1, m, frames }));
        return Engine.Reshape(wave, new[] { wave.Length });
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids (ending in the end token); the result is the 16 kHz speech
    /// in the voice's x-vector: the generated spectrogram through HiFi-GAN.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var ids = TokenIds(input, nameof(input));
        if (ids.Length == 0) throw new ArgumentException("There are no text tokens.", nameof(input));
        var voice = Voice ?? throw new InvalidOperationException($"{GetType().Name} speaks in a given voice; set {nameof(Voice)} first (see CreateVoice).");
        return Vocode(GenerateSpectrogram(ids, voice));
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
            Name = _useNativeMode ? "SpeechT5-Native" : "SpeechT5-ONNX",
            Description = "SpeechT5: unified-modal encoder-decoder pre-training, fine-tuned for text to speech (Ao et al., 2022)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "SpeechT5";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Stage"] = _options.CurrentStage.ToString();
        metadata.AdditionalInfo["StopCriterion"] = _options.StopCriterion.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SpeechT5<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
