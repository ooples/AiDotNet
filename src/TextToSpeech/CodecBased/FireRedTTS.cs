using System.Runtime.CompilerServices;
using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
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
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// FireRedTTS (Guo et al. 2024): a foundation text-to-speech system — a decoder-only language model predicts a
/// recording's semantic tokens from text and a speaker embedding, a flow-matching Mel decoder turns the tokens into a
/// 16 kHz mel in a prompt's timbre, and BigVGAN-v2 makes the audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The parts (§3): the semantic-aware speech tokenizer (HuBERT features, a convolutional encoder and a product
/// quantizer of 16,384 tokens at 40 ms, <see cref="FireRedTtsSpeechTokenizer{T}"/>) and its acoustic encoder (an
/// ECAPA-TDNN speaker embedding); the GPT (§3.2, the release's XTTS-style model, <see cref="FireRedTtsGpt{T}"/>); the
/// flow-matching Mel decoder (§3.3.1, <see cref="FireRedTtsFlow{T}"/>); and BigVGAN-v2 at 24 kHz (the release) or the
/// paper's 48 kHz (<see cref="FireRedTTSOptions.Vocoder"/>).</para>
/// <para>Synthesis runs the release's <c>synthesize</c>: the text split (<c>text_split</c>), each piece normalized and
/// tokenized, seven token sequences sampled and the second shortest kept, the flow and the vocoder, the pieces joined.
/// Instruction tuning (§4.2) is opt-in (<see cref="FireRedTTSOptions.InstructionTuning"/>): an emotion and paralinguistic
/// markup (<see cref="SynthesizeInstruction"/>).</para>
/// <para><b>For Beginners:</b> give FireRedTTS a few seconds of someone speaking and it reads new text in that voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1);
/// var model = new FireRedTTS&lt;float&gt;(architecture, new FireRedTTSOptions());
/// model.LoadReferenceWeights("FireRedTTS");          // fireredtts_gpt.pt, fireredtts_token2wav.pt, fireredtts_speaker.bin
/// model.Voice = model.CreateVoice(promptAudio, promptSampleRate);
/// var audio = model.Synthesize("小红书，是中国大陆的网络购物和社交平台。");   // 24 kHz
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "FireRedTTS: A Foundation Text-To-Speech Framework for Industry-Level Generative Speech Applications",
    "https://arxiv.org/abs/2409.03283", Year = 2024,
    Authors = "Hao-Han Guo, Yao Hu, Kun Liu, Fei-Yu Shen, Xu Tang, Yi-Chen Wu, Feng-Long Xie, Kun Xie, Kai-Tuo Xu")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "language model", LearningRate = 5e-6, Beta1 = 0.9, Beta2 = 0.96, Epsilon = 1e-8,
    WeightDecay = 0.01, Provenance = RecipeProvenance.ReleasedCode,
    Source = "The paper gives no optimizer. The release's GPT is XTTS's (fireredtts/modules/gpt/gpt.py: 'ported from coqui-ai/TTS'), "
             + "whose recipe (recipes/ljspeech/xtts_v2/train_gpt_xtts.py) trains with AdamW at 5e-6, betas (0.9, 0.96), eps 1e-8, "
             + "weight decay 0.01 on weights only, MultiStepLR halving at steps 900k, 2.7M and 5.4M, and losses 0.01 x text CE + 1.0 x mel CE.")]
[PaperOptimizer(OptimizerKind.Adam, Component = "flow", LearningRate = 1e-3, WarmupSteps = 2500,
    Schedule = LearningRateSchedulerType.NoamHoldAnnealing, HoldFraction = 0, DecayRate = 0.5, MaxGradientNorm = 5,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "The paper gives the flow's loss (eq. 6) and 20% condition dropout but no optimizer; its estimator is CosyVoice's "
             + "ConditionalDecoder, trained in FunAudioLLM/CosyVoice's cosyvoice.yaml with Adam at 1e-3 under WarmupLR "
             + "(2,500 steps, then inverse-square-root decay) and clipping at 5.")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "speech tokenizer", LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.01,
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "The paper gives eq. 1's weights (1, 1000, 1) and 300k iterations; its tokenizer follows DPQ-VAE ([24], Guo et al. 2024), "
             + "trained with AdamW (0.9, 0.95) and EMA codebooks; neither states the rate (1e-4 chosen) or the decay (PyTorch's AdamW "
             + "default 0.01).")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "acoustic language model", LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.1,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 4000, PostWarmupDecay = LinearWarmupScheduler.DecayMode.Cosine,
    MaxGradientNorm = 1.0, Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "The paper gives no recipe for its multi-stream LM, which follows MusicGen ([34]); MusicGen's (audiocraft "
             + "config/solver/musicgen/default.yaml): AdamW at 1e-4, betas (0.9, 0.95), weight decay 0.1, cosine after 4,000 "
             + "warm-up steps, clipping at 1.")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "mel codec", LearningRate = 1e-4, Beta1 = 0.8, Beta2 = 0.99,
    Schedule = LearningRateSchedulerType.Exponential, DecayRate = 0.999996, MaxGradientNorm = 1000,
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Neither the paper nor SoCodec ([33]) releases its codec's training; DAC's (descript-audio-codec conf/base.yml): AdamW "
             + "at 1e-4, betas (0.8, 0.99), ExponentialLR 0.999996, clipping at 1e3, for the codec and its discriminator.")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "mel discriminator", LearningRate = 1e-4, Beta1 = 0.8, Beta2 = 0.99,
    Schedule = LearningRateSchedulerType.Exponential, DecayRate = 0.999996, MaxGradientNorm = 1000,
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "DAC's discriminator recipe, as for the codec.")]
public partial class FireRedTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly FireRedTTSOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private FireRedTtsGpt<T>? _gpt;
    private IndexTtsEcapa<T>? _speaker;
    private FireRedTtsFlow<T>? _flow;
    private BigVganGenerator<T>? _vocoder;
    private FireRedTtsSpeechTokenizer<T>? _tokenizer;
    private FireRedTtsMelCodec<T>? _codec;
    private FireRedTtsAcousticLm<T>? _acousticLm;
    private MegaTtsMelDiscriminator<T>? _discriminator;
    private readonly List<LayerBase<T>> _codecLayers = new();
    private readonly List<LayerBase<T>> _acousticLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private readonly List<LayerBase<T>> _gptLayers = new();
    private readonly List<LayerBase<T>> _speakerLayers = new();
    private readonly List<LayerBase<T>> _flowLayers = new();
    private readonly List<LayerBase<T>> _hubertLayers = new();
    private readonly List<LayerBase<T>> _tokenizerLayers = new();
    private DifferentiableMel<T>? _speakerMel;
    private DifferentiableMel<T>? _flowMel;

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The rate the tokenizer, the speaker encoder and the mels read (fixed by the paper and release).</summary>
    public const int AnalysisRate = 16000;

    /// <summary>Creates an ONNX-backed FireRedTTS for inference.</summary>
    public FireRedTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, FireRedTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FireRedTTSOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable FireRedTTS.</summary>
    public FireRedTTS(NeuralNetworkArchitecture<T> architecture, FireRedTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FireRedTTSOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.HiddenDim % o.NumHeads != 0) throw new ArgumentException("HiddenDim must divide by NumHeads.");
        if (o.SpeakerDim != o.TokenizerDim) throw new ArgumentException("The tokenizer's decoder adds the speaker embedding: SpeakerDim must equal TokenizerDim.");
        if (o.AudioTokens != o.CodebookSize * o.CodebookSize + 2)
            throw new ArgumentException("AudioTokens holds the tokenizer's CodebookSize² tokens and the start and stop tokens.");
        if (o.VocoderUpsampleRates.Aggregate(1, (a, b) => a * b) * AnalysisRate != o.SampleRate * 160)
            throw new ArgumentException("The vocoder's upsampling must take the 10 ms mel to SampleRate.");
        if (o.HopSize * AnalysisRate != o.SampleRate * 640) throw new ArgumentException("HopSize is the samples of one 40 ms token at SampleRate.");
        if (o.MaxTextLength >= o.MaxTextTokens) throw new ArgumentException("MaxTextLength must stay below MaxTextTokens.");
        if (o.Candidates < 1 || o.CandidateRank < 0) throw new ArgumentException("At least one candidate, and a rank from 0.");
        if (o.CodecDim != o.SpeakerDim) throw new ArgumentException("The Mel codec's decoder adds the speaker embedding: CodecDim must equal SpeakerDim.");
        if (o.CodecStrides.Aggregate(1, (a, b) => a * b) != 2) throw new ArgumentException("The Mel codec takes the 10 ms mel to 20 ms frames: its strides multiply to 2.");
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        // XTTS's GPT recipe does not clip the gradient; the flow's recipe declares its own clipping.
        MaxGradNorm = NumOps.Zero;
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>Synthesis imitates a recording of the speaker (<see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool HasPaperLayers => _gpt is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal FireRedTtsGpt<T> Gpt => _gpt ?? throw NotNative();
    internal IndexTtsEcapa<T> SpeakerEncoder => _speaker ?? throw NotNative();
    internal FireRedTtsFlow<T> Flow => _flow ?? throw NotNative();
    internal BigVganGenerator<T> Vocoder => _vocoder ?? throw NotNative();
    internal FireRedTtsSpeechTokenizer<T> SpeechTokenizer => _tokenizer ?? throw NotNative();
    internal FireRedTtsMelCodec<T> MelCodec => _codec ?? throw NotNative();
    internal FireRedTtsAcousticLm<T> AcousticLm => _acousticLm ?? throw NotNative();
    internal MegaTtsMelDiscriminator<T> MelDiscriminator => _discriminator ?? throw NotNative();

    /// <summary>The text tokenizer (the release's, with the paper's instruction tags when instruction tuning).</summary>
    internal FireRedTtsTokenizer TextTokenizer => FireRedTtsTokenizer.Get(_options.InstructionTuning);

    private int TextTokens => Math.Max(_options.VocabSize, _options.InstructionTuning ? TextTokenizer.Encoding.VocabularySize : 0);

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(TextTokens) : base.GetInputDomain(inputShape);

    internal FireRedTtsGptConfiguration GptConfiguration()
    {
        var o = _options;
        return new FireRedTtsGptConfiguration(o.HiddenDim, o.NumHeads, o.NumDecoderLayers, o.MaxTextTokens, o.MaxAudioTokens, TextTokens,
            o.AudioTokens, o.StartTextToken, o.StopTextToken, o.AudioTokens - 2, o.AudioTokens - 1, o.SpeakerDim, o.ReferenceHidden,
            o.DropoutRate, o.InstructionTuning ? 4 : 0, o.InstructionTuning ? FireRedTtsInstruction.LabelCount : 0);
    }

    internal FireRedTtsFlowConfiguration FlowConfiguration()
    {
        var o = _options;
        return new FireRedTtsFlowConfiguration(o.MelChannels, o.CodebookSize, o.CodebookDim, o.FlowDim, o.FlowHeads, o.FlowLinearUnits, o.FlowBlocks,
            o.FlowCrossAttentionBlocks, o.FlowDropout, o.FlowRegulatorBlocks, 4, o.PromptEncoderHidden, o.PromptEncoderReduction,
            o.PromptEncoderLayers, o.PromptEncoderBlocks, o.PromptEncoderKernel, o.FlowDecoderChannels, o.FlowDecoderHeads, o.FlowDecoderHeadDim,
            o.FlowDecoderBlocksPerLevel, o.FlowDecoderMidBlocks, o.InferenceCfgRate, o.TrainingCfgRate, 1e-6);
    }

    internal FireRedTtsTokenizerConfiguration TokenizerConfiguration()
    {
        var o = _options;
        return new FireRedTtsTokenizerConfiguration(o.SslDim, o.TokenizerDim, o.TokenizerDownsampleStrides, o.CodebookSize, o.CodebookDim, 2, 80,
            o.TokenizerUpsampleStrides, 640, 320, o.CodebookEmaDecay, o.DeadCodeThreshold, 1.0, o.HubertNormalizeWave);
    }

    internal HubertConfiguration HubertConfiguration()
    {
        var o = _options;
        return new HubertConfiguration
        {
            ConvDims = o.HubertConvDims, HiddenSize = o.HubertDim, Layers = o.HubertLayers, Heads = o.HubertHeads,
            IntermediateSize = o.HubertIntermediate, PositionalConvKernel = o.HubertPositionalKernel, PositionalConvGroups = o.HubertPositionalGroups,
        };
    }

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
        if (o.SslDim != o.HubertDim) throw new ArgumentException("The tokenizer reads HuBERT's features: SslDim must equal HubertDim.");
        _gpt = new FireRedTtsGpt<T>(Engine, _gptLayers, GptConfiguration(), random);
        _speaker = new IndexTtsEcapa<T>(Engine, _speakerLayers, new IndexTtsEcapaConfiguration(80, o.SpeakerDim, o.SpeakerChannels, o.SpeakerKernels,
            o.SpeakerDilations, o.SpeakerAttentionChannels, o.SpeakerSeChannels, o.SpeakerRes2NetScale), random);
        _tokenizer = new FireRedTtsSpeechTokenizer<T>(Engine, _hubertLayers, _tokenizerLayers, TokenizerConfiguration(), HubertConfiguration(), random);
        // The flow's input embedding reads the tokenizer's codebooks, frozen.
        _flow = new FireRedTtsFlow<T>(Engine, _flowLayers, FlowConfiguration(), b => SpeechTokenizer.Quantizer.Codebooks[b], random);
        _vocoder = new BigVganGenerator<T>(Engine, random, o.MelChannels, o.VocoderInitialChannels, o.VocoderUpsampleRates, o.VocoderUpsampleKernels,
            o.VocoderResBlockKernels, o.VocoderResBlockDilations, snakeBeta: true, tanhAtFinal: false, biasAtFinal: false);
        _codec = new FireRedTtsMelCodec<T>(Engine, _codecLayers, new FireRedTtsMelCodecConfiguration(o.MelChannels, o.CodecDim, o.CodecStrides,
            o.CodebookSize, o.CodecCodebookDim, o.CodecStreams, o.CodecConvGroups, o.CodecEmaDecay, o.CodecCommitmentWeight), random);
        _acousticLm = new FireRedTtsAcousticLm<T>(Engine, _acousticLayers, new FireRedTtsAcousticLmConfiguration(o.CodecStreams,
            o.CodebookSize * o.CodebookSize, o.CodebookSize * o.CodebookSize, o.AcousticLmDim, o.AcousticLmHeads, o.AcousticLmLayers,
            o.AcousticLmFeedForward, o.AcousticLmDropout, Enumerable.Range(0, o.CodecStreams).ToArray(), o.Lookahead, o.SpeakerDim,
            o.ReferenceHidden), random);
        _discriminator = new MegaTtsMelDiscriminator<T>(Engine, _discriminatorLayers, o.DiscriminatorWindows, o.MelChannels, o.DiscriminatorHidden);
        // The GPT is the model's main network; the others are components, each trained in its own stage (the vocoder is
        // pretrained and frozen: its GAN stage is not ported).
        AddUnsplitLayers(_gptLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_speakerLayers);
        ComponentLayers.AddRange(_flowLayers);
        ComponentLayers.AddRange(_hubertLayers);
        ComponentLayers.AddRange(_tokenizerLayers);
        ComponentLayers.AddRange(_vocoder.Layers);
        ComponentLayers.AddRange(_codecLayers);
        ComponentLayers.AddRange(_acousticLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
    }

    // ---------------------------------------------------------------- text

    /// <summary>The release's text pieces (<c>text_split</c>; its dropped last piece kept unless
    /// <see cref="FireRedTTSOptions.ReleaseTextSplit"/>).</summary>
    public IReadOnlyList<string> SplitText(string text) =>
        FireRedTtsTextSplitter.Split(text ?? throw new ArgumentNullException(nameof(text)), _options.ReleaseTextSplit);

    /// <summary>The token ids of a text piece (normalized, language-tagged, tokenized; <c>VoiceBpeTokenizer.encode</c>).</summary>
    /// <param name="language">"zh", "en" or "auto" (detected).</param>
    public int[] TokenizeText(string text, string language = "auto") =>
        TextTokenizer.Encode(text ?? throw new ArgumentNullException(nameof(text)), language);

    /// <inheritdoc />
    /// <remarks>The text's token ids (<see cref="TokenizeText"/>) as one piece.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        return IndexTtsOps.Ids<T>(TokenizeText(text));
    }

    private int[] TextIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= TextTokens) throw new ArgumentOutOfRangeException(name, $"Text token ids lie in [0, {TextTokens}).");
        }
        return ids;
    }

    // ---------------------------------------------------------------- audio features

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    /// <summary>A recording at <paramref name="rate"/> resampled to 16 kHz by torchaudio's default (<c>load_audio</c>) and
    /// clipped to [−1, 1].</summary>
    internal double[] To16k(Tensor<T> audio, int rate)
    {
        var resampled = rate == AnalysisRate ? audio : AudioHelper<T>.ResampleBandLimited(Engine, Wave(Doubles(audio)), rate, AnalysisRate);
        return Doubles(resampled).Select(v => Math.Max(-1.0, Math.Min(1.0, v))).ToArray();
    }

    /// <summary>
    /// The flow's 100-band mel <c>[frames, 100]</c> of a recording at <paramref name="rate"/>
    /// (<c>MelSpectrogramExtractor</c>): resampled to 16 kHz with torchaudio's "kaiser_best" settings (a 16 kHz recording
    /// up to 24 kHz and back first), then <c>mel_spectrogram</c> (n_fft 1024, hop 160, window 1024, natural log).
    /// </summary>
    internal Tensor<T> FlowMel(Tensor<T> audio, int rate)
    {
        var wave = Wave(Doubles(audio));
        wave = rate == AnalysisRate
            ? AudioHelper<T>.ResampleKaiser(Engine, AudioHelper<T>.ResampleKaiser(Engine, wave, rate, 24000), 24000, AnalysisRate)
            : AudioHelper<T>.ResampleKaiser(Engine, wave, rate, AnalysisRate);
        _flowMel ??= new DifferentiableMel<T>(Engine, AnalysisRate, 1024, 160, 1024, _options.MelChannels, 0, AnalysisRate / 2.0, singlePrecisionBasis: true);
        return _flowMel.Forward(wave);
    }

    /// <summary>The speaker encoder's and tokenizer's 80-band mel <c>[frames, 80]</c> of 16 kHz audio
    /// (<c>TorchMelSpectrogram</c>: n_fft 1024, hop 160, window 640, up to 8 kHz).</summary>
    internal Tensor<T> SpeakerMel(double[] audio16k)
    {
        _speakerMel ??= new DifferentiableMel<T>(Engine, AnalysisRate, 1024, 160, 640, 80, 0, 8000, singlePrecisionBasis: true);
        return _speakerMel.Forward(Wave(audio16k));
    }

    /// <summary>The ECAPA-TDNN speaker embedding <c>[SpeakerDim]</c> of 16 kHz audio (<c>SpeakerEmbedddingExtractor</c>).</summary>
    internal Tensor<T> SpeakerEmbedding(double[] audio16k)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _speakerLayers) layer.SetTrainingMode(false);
        return SpeakerEncoder.Forward(SpeakerMel(audio16k));
    }

    /// <summary>A reader of ECAPA-TDNN names that maps the release's <c>Fp32BatchNorm</c> (<c>….norm.bn.*</c>).</summary>
    internal static Func<string, int[], double[]> EcapaReader(Func<string, int[], double[]> read) => (name, shape) =>
    {
        foreach (var field in new[] { "weight", "bias", "running_mean", "running_var" })
            if (name.EndsWith("norm." + field, StringComparison.Ordinal))
                return read(name.Substring(0, name.Length - field.Length) + "bn." + field, shape);
        return read(name, shape);
    };

    // ---------------------------------------------------------------- codec tokens

    /// <inheritdoc />
    public int NumCodebooks => 1;

    /// <inheritdoc />
    /// <remarks>The tokenizer's 16,384 semantic tokens (two sub-codebooks of 128).</remarks>
    public int CodebookSize => _options.CodebookSize * _options.CodebookSize;

    /// <inheritdoc />
    /// <remarks>25 tokens a second (40 ms).</remarks>
    public int CodecFrameRate => AnalysisRate / 640;

    /// <summary>The semantic tokens of a recording at <paramref name="rate"/> (<c>extract_speech_tokens</c>).</summary>
    public int[] EncodeSemanticTokens(Tensor<T> audio, int rate)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (!HasPaperLayers) throw NotNative();
        using var _ = new NoGradScope<T>();
        return SpeechTokenizer.Tokenize(To16k(audio, rate));
    }

    /// <inheritdoc />
    /// <remarks>The semantic tokens <c>[frames, 1]</c> of audio at the model's rate.</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        var ids = EncodeSemanticTokens(audio, _options.SampleRate);
        var tokens = new Tensor<T>(new[] { ids.Length, 1 });
        for (int i = 0; i < ids.Length; i++) tokens[i, 0] = Ops.FromDouble(ids[i]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>The audio of semantic tokens <c>[frames, 1]</c> (or <c>[frames]</c>) in the voice's timbre: the flow
    /// with the voice's prompt mel, then the vocoder.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (!HasPaperLayers) throw NotNative();
        if (tokens.Rank == 2 && tokens.Shape[1] != 1) throw new ArgumentException("Semantic tokens are [frames, 1].", nameof(tokens));
        var ids = Enumerable.Range(0, tokens.Shape[0]).Select(i => (int)Math.Round(Ops.ToDouble(tokens.GetFlat(i)))).ToArray();
        var prompt = Prompt(RequireVoice());
        using var _ = new NoGradScope<T>();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        return Wave(_options.MelDecoder == FireRedTtsMelDecoder.Streamable ? DecodeStreamable(ids, prompt.Speaker, random) : Decode(ids, prompt.Mel, random));
    }

    // ---------------------------------------------------------------- voices

    private sealed class PromptInfo
    {
        public Tensor<T> Speaker = null!;       // [SpeakerDim]
        public Tensor<T> Mel = null!;           // [frames, 100], the flow's prompt
    }

    private readonly ConditionalWeakTable<TtsVoice<T>, PromptInfo> _prompts = new();

    /// <summary>A voice from a recording of the speaker at <paramref name="sampleRate"/>: its speaker embedding (16 kHz,
    /// <c>extract_spk_embeddings</c>) and the flow's prompt mel (<c>MelSpectrogramExtractor</c>).</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (!HasPaperLayers) throw NotNative();
        var audio = sampleRate == _options.SampleRate ? recording : AudioHelper<T>.ResampleBandLimited(Engine, Wave(Doubles(recording)), sampleRate, _options.SampleRate);
        var voice = new TtsVoice<T> { ReferenceAudio = audio };
        _prompts.Add(voice, PromptOf(recording, sampleRate));
        return voice;
    }

    private PromptInfo PromptOf(Tensor<T> audio, int rate)
    {
        using var _ = new NoGradScope<T>();
        return new PromptInfo { Speaker = SpeakerEmbedding(To16k(audio, rate)), Mel = FlowMel(audio, rate) };
    }

    private PromptInfo Prompt(TtsVoice<T> voice)
    {
        if (_prompts.TryGetValue(voice, out var info)) return info;
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} imitates a recording; set ReferenceAudio on the voice.");
        info = PromptOf(audio, _options.SampleRate);
        _prompts.Add(voice, info);
        return info;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>Speech at <see cref="FireRedTTSOptions.SampleRate"/> in the voice's speaker (the release's
    /// <c>synthesize</c>).</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        var pieces = SplitText(text);
        if (pieces.Count == 0) throw new ArgumentException("The text has nothing to speak.", nameof(text));
        return Run(random => pieces.Select(piece => (TokenizeText(piece), (int[]?)null)).ToList(), null);
    }

    /// <summary>
    /// Instruction-tuned speech (§4.2): the text with the paper's paralinguistic markup (Table 1: tags such as
    /// <c>[laugh]</c> and <c>[breath]</c> in the text, and after a Chinese character <c>ˆ</c> for speaking with a laugh,
    /// <c>@</c> for emphasis, <c>P</c>, <c>C</c>, <c>R</c> or <c>S</c> for a filled pause, confirmation, realization or
    /// surprise) in the given emotion.
    /// </summary>
    public Tensor<T> SynthesizeInstruction(string text, FireRedTtsEmotion emotion)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (!_options.InstructionTuning) throw new InvalidOperationException("Enable FireRedTTSOptions.InstructionTuning.");
        if (!HasPaperLayers) throw NotNative();
        var pieces = SplitText(FireRedTtsInstruction.Strip(text));
        var marked = FireRedTtsInstruction.Parse(text);
        return Run(random => pieces.Select(piece => FireRedTtsInstruction.Encode(TextTokenizer, piece, marked)).ToList(), (int)emotion);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids, spoken as one piece.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var ids = TextIds(input, nameof(input));
        if (ids.Length == 0) throw new ArgumentException("There are no text tokens.", nameof(input));
        return Run(_ => new List<(int[], int[]?)> { (ids, null) }, null);
    }

    internal FireRedTtsDecoding Decoding()
    {
        var o = _options;
        return new FireRedTtsDecoding(o.Sample, o.TopK, o.TopP, o.Temperature, o.RepetitionPenalty, o.Candidates, o.CandidateRank, o.MaxNewTokens,
            o.ConstrainToSemanticTokens);
    }

    private Tensor<T> Run(Func<Random, List<(int[] Ids, int[]? Labels)>> pieces, int? emotion)
    {
        var o = _options;
        if (!(o.TopP > 0 && o.TopP <= 1)) throw new InvalidOperationException("TopP must be in (0, 1].");
        if (!(o.Temperature > 0)) throw new InvalidOperationException("Temperature must be positive.");
        var voice = RequireVoice();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
            var prompt = Prompt(voice);
            var prefix = Gpt.Prefix(prompt.Speaker, emotion);
            var wave = new List<double>();
            foreach (var (ids, labels) in pieces(random))
            {
                if (ids.Length >= o.MaxTextLength) throw new ArgumentException($"A piece has {ids.Length} tokens; the release allows fewer than {o.MaxTextLength}.");
                var candidates = Gpt.Generate(prefix, ids, labels, Decoding(), random);
                var tokens = FireRedTtsGpt<T>.Choose(candidates, o.CandidateRank);
                if (tokens.Length == 0) continue;
                wave.AddRange(o.MelDecoder == FireRedTtsMelDecoder.Streamable
                    ? DecodeStreamable(tokens, prompt.Speaker, random)
                    : Decode(tokens, prompt.Mel, random));
            }
            return Wave(wave.ToArray());
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary><c>Token2Wav.inference</c>: the flow's mel of the tokens (noise from <paramref name="random"/>,
    /// <see cref="FireRedTTSOptions.FlowSteps"/> steps) and the vocoder's audio.</summary>
    internal double[] Decode(IReadOnlyList<int> tokens, Tensor<T> promptMel, Random random)
    {
        var o = _options;
        int frames = 4 * tokens.Count;
        var noise = new Tensor<T>(new[] { frames, o.MelChannels }, TorchParameters.Gaussian<T>(random, frames * o.MelChannels, 1.0));
        var mel = Flow.Inference(tokens, promptMel, o.FlowSteps, noise);
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, o.MelChannels, frames }));
        return Doubles(wave);
    }

    /// <summary>The streamable decoder (§3.3.2): the acoustic LM's four streams for the tokens (two frames each, sampled
    /// with top-k and temperature), the Mel codec's mel of them in the speaker's voice, and the vocoder's audio.</summary>
    internal double[] DecodeStreamable(IReadOnlyList<int> tokens, Tensor<T> speaker, Random random)
    {
        var o = _options;
        var streams = AcousticLm.Generate(tokens, speaker, logits => SampleTopK(logits, o.AcousticTopK, o.AcousticTemperature, random));
        var mel = MelCodec.Detokenize(streams, speaker);
        int frames = mel.Shape[0];
        return Doubles(Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, o.MelChannels, frames })));
    }

    // audiocraft's sample_top_k: the k most likely, the temperature-scaled softmax over them, one draw.
    internal static int SampleTopK(double[] logits, int k, double temperature, Random random)
    {
        if (!(temperature > 0)) throw new InvalidOperationException("AcousticTemperature must be positive.");
        var order = Enumerable.Range(0, logits.Length).OrderByDescending(v => logits[v]).ThenBy(v => v).Take(Math.Max(1, Math.Min(k, logits.Length))).ToArray();
        double max = logits[order[0]], sum = 0;
        var p = order.Select(v => Math.Exp((logits[v] - max) / temperature)).ToArray();
        foreach (double v in p) sum += v;
        double u = random.NextDouble() * sum, acc = 0;
        for (int i = 0; i < order.Length; i++)
        {
            acc += p[i];
            if (u < acc) return order[i];
        }
        return order[order.Length - 1];
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
            Name = _useNativeMode ? "FireRedTTS-Native" : "FireRedTTS-ONNX",
            Description = "FireRedTTS: a GPT over semantic tokens, a flow-matching Mel decoder and BigVGAN-v2 (Guo et al., 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "FireRedTTS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FireRedTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
