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
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>A GLM-4-Voice reply: its text, its speech tokens and its 22.05 kHz audio.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class GlmVoiceReply<T>
{
    /// <summary>The reply's text tokens, decoded.</summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>The reply's speech tokens (codes of the speech tokenizer).</summary>
    public int[] SpeechTokens { get; init; } = Array.Empty<int>();

    /// <summary>The speech decoded to a waveform; empty when the reply has no speech tokens.</summary>
    public Tensor<T> Audio { get; init; } = new(new[] { 0 });
}

/// <summary>
/// GLM-4-Voice (Zeng et al. 2024): an end-to-end spoken chatbot. A 12.5 Hz supervised speech tokenizer turns speech into
/// tokens; GLM-4-9B, extended by those tokens and pre-trained on a trillion speech-text tokens, replies in text and
/// speech tokens interleaved; a flow-matching decoder streams the speech tokens to audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Speech tokenizer</b> (§3.1): whisper-large-v3's encoder with causal convolutions and block-causal
/// attention, average-pooled 4× and vector-quantized (16384 codes) after its 16th layer; fine-tuned on ASR so the codes
/// keep the semantics.</para>
/// <para><b>Language model</b> (§4): ChatGLM's decoder (RMSNorm, multi-query attention with rotary positions on half
/// of each head, SwiGLU), its vocabulary grown by the speech tokens. Replies use "streaming thoughts" (§3.3): 13 text
/// tokens, then 26 speech tokens, alternating, so speech can start before the text is finished.</para>
/// <para><b>Speech decoder</b> (§3.2): CosyVoice's flow matching with a Conformer that attends in 10-token blocks,
/// decoding 0.8 s blocks with what came before as the prompt, then HiFT.</para>
/// <para><see cref="Synthesize"/> reads a text aloud (by default the paper's speech-and-text-to-speech subtask, the text
/// fed as the reply's text); <see cref="Chat(string)"/> and <see cref="Chat(Tensor{T})"/> answer a text or a 16 kHz
/// recording.</para>
/// <para><b>For Beginners:</b> GLM-4-Voice listens and talks. Your speech becomes "speech words", a large language
/// model writes its answer as a mix of normal words and speech words, and a decoder turns the speech words into
/// sound.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new GLM4Voice&lt;float&gt;(architecture, new GLM4VoiceOptions { TokenizerDirectory = "glm-4-voice-9b" });
/// model.LoadLanguageModel("glm-4-voice-9b");
/// model.LoadSpeechTokenizer("glm-4-voice-tokenizer");
/// model.LoadSpeechDecoder("glm-4-voice-decoder");
/// var audio = model.Synthesize("Hello world.");
/// var reply = model.Chat(recording16kHz);           // reply.Text, reply.Audio
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot",
    "https://arxiv.org/abs/2412.02612",
    Year = 2024,
    Authors = "Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, Jie Tang"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.1,
                Source = "Zeng et al. 2024 §4.1.1: AdamW with beta1 0.9 and beta2 0.95; §4.2.2: fine-tuning decays the rate "
                        + "from 1e-5 to 1e-6 with weight decay 0.1, dropout 0.5 and clipping at 1.0 (pre-training decays it "
                        + "from 6e-5 to 6e-6).")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "speech tokenizer", LearningRate = 1e-5,
                Source = "Zeng et al. 2024a App. B.1 (the tokenizer GLM-4-Voice §3.1 adopts): fine-tuned from whisper-large-v3 for 2 "
                        + "epochs at batch 4096 and a learning rate of 1e-5. The optimizer is not named; AdamW, the papers' "
                        + "optimizer elsewhere, with its default moments.")]
[PaperOptimizer(OptimizerKind.Adam, Component = "speech decoder", LearningRate = 1e-3, WarmupSteps = 2500,
    Schedule = LearningRateSchedulerType.NoamHoldAnnealing, HoldFraction = 0, DecayRate = 0.5, MaxGradientNorm = 5,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "Zeng et al. 2024a App. B.2: the flow trained from scratch at a learning rate of 1e-3. The rest is CosyVoice's "
        + "released recipe for the same model (examples/libritts/cosyvoice/conf/cosyvoice.yaml): Adam under WarmupLR, "
        + "2,500 warmup steps then inverse-square-root decay, clipping at 5.")]
public partial class GLM4Voice<T> : TtsModelBase<T>
{
    private readonly GLM4VoiceOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private GlmDecoder<T>? _lm;
    private GlmSpeechTokenizer<T>? _speechTokenizer;
    private GlmVoiceDecoder<T>? _speechDecoder;
    private GlmTokenizer? _tokenizer;
    private Random? _samplingRandom;
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _speechTokenizerLayers = new();
    private readonly List<LayerBase<T>> _flowLayers = new();
    private readonly List<LayerBase<T>> _hiftLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private const int SpeechSampleRate = 16000;

    // The options' seed; 42 when unset.
    private int SeedValue => _options.Seed ?? 42;

    /// <summary>Creates an ONNX-backed GLM-4-Voice for inference.</summary>
    public GLM4Voice(NeuralNetworkArchitecture<T> architecture, string modelPath, GLM4VoiceOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new GLM4VoiceOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable GLM-4-Voice.</summary>
    public GLM4Voice(NeuralNetworkArchitecture<T> architecture, GLM4VoiceOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new GLM4VoiceOptions();
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
        if (o.HiddenDim <= 0 || o.Layers <= 0 || o.NumHeads <= 0 || o.HeadDim <= 0 || o.FeedForwardDim <= 0)
            throw new ArgumentException("The language model's sizes must be positive.");
        if (o.KeyValueGroups <= 0 || o.NumHeads % o.KeyValueGroups != 0)
            throw new ArgumentException("The attention heads must divide into the key-value groups.");
        if (o.SampleRate != 22050 || o.HopSize != 256 || o.MelChannels != 80)
            throw new ArgumentException("The speech decoder renders 80-band mels at 22.05 kHz with a hop of 256 (HiFT's 8·8·4).");
        if (o.TextTokensPerChunk <= 0 || o.SpeechTokensPerChunk <= 0) throw new ArgumentException("Streaming-thoughts chunks must be positive.");
        if (o.Temperature <= 0 || o.TopP <= 0 || o.TopP > 1) throw new ArgumentException("Sampling needs a positive temperature and a top-p in (0, 1].");
        if (o.MaxNewTokens <= 0) throw new ArgumentException("MaxNewTokens must be positive.");
    }

    /// <summary>The longest token sequence the language model takes (8192).</summary>
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>The release's decoder speaks with one voice: its flow model takes no speaker embedding.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.None;

    private bool HasPaperLayers => _lm is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal GlmDecoder<T> LanguageModel => _lm ?? throw NotNative();
    internal GlmSpeechTokenizer<T> SpeechTokenizer => _speechTokenizer ?? throw NotNative();
    internal GlmVoiceDecoder<T> SpeechDecoder => _speechDecoder ?? throw NotNative();

    /// <inheritdoc />
    /// <remarks>Text token ids: those below the first speech token (all of the language model's rows without a
    /// tokenizer).</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape)
    {
        if (!HasPaperLayers) return base.GetInputDomain(inputShape);
        return LayerInputDomain.Indices(_options.TokenizerDirectory is null && _tokenizer is null ? _options.VocabSize : AudioOffset);
    }

    internal GlmConfiguration LanguageModelConfiguration(double dropout) => new()
    {
        VocabSize = _options.VocabSize, Hidden = _options.HiddenDim, Layers = _options.Layers, Heads = _options.NumHeads,
        KeyValueGroups = _options.KeyValueGroups, HeadDim = _options.HeadDim, FeedForward = _options.FeedForwardDim,
        NormEpsilon = _options.NormEpsilon, RopeRatio = _options.RopeRatio, Dropout = dropout,
    };

    internal GlmSpeechTokenizerConfiguration SpeechTokenizerConfiguration => new()
    {
        Mels = 128, Dim = _options.SpeechTokenizerDim, Heads = _options.SpeechTokenizerHeads, FeedForward = _options.SpeechTokenizerFeedForwardDim,
        EncoderLayers = _options.SpeechTokenizerLayers, MaxSourcePositions = _options.SpeechTokenizerMaxPositions, PoolingKernel = _options.PoolingKernel,
        PoolingPosition = _options.PoolingPosition, QuantizePosition = _options.QuantizePosition, CodebookSize = _options.SpeechCodebookSize,
        CausalBlockSize = _options.CausalBlockSize, EmaDecay = _options.EmaDecay, CommitCoefficient = _options.CommitCoefficient,
        LossScale = _options.QuantizeLossScale, RestartInterval = _options.RestartInterval, DecoderLayers = _options.AsrDecoderLayers,
        Vocabulary = _options.AsrVocabulary, MaxTargetPositions = _options.AsrMaxTargetPositions,
    };

    internal GlmVoiceDecoderConfiguration SpeechDecoderConfiguration
    {
        get
        {
            var o = _options;
            var release = GlmVoiceDecoderConfiguration.Release(o.FlowLoss);
            var flow = release.Flow with
            {
                VocabSize = o.SpeechCodebookSize, InputSize = o.FlowDim,
                Encoder = release.Flow.Encoder with
                {
                    InputSize = o.FlowDim, OutputSize = o.FlowDim, Heads = o.FlowHeads, LinearUnits = o.FlowFeedForwardDim, Blocks = o.FlowLayers,
                    StaticChunkSize = o.FlowBlockTokens,
                },
                DecoderChannels = (int[])o.DecoderChannels.Clone(), DecoderHeads = o.DecoderHeads, DecoderHeadDim = o.DecoderHeadDim,
                DecoderBlocksPerLevel = o.DecoderBlocks, DecoderMidBlocks = o.DecoderMidBlocks, Timesteps = o.FlowTimesteps,
                InferenceCfgRate = o.InferenceCfgRate, TrainingCfgRate = o.TrainingCfgRate,
            };
            var hift = release.Hift with { BaseChannels = o.HiftBaseChannels, F0Channels = o.HiftF0Channels };
            return new GlmVoiceDecoderConfiguration(flow, hift, BlockTokens: o.FlowBlockTokens, NoiseSeed: SeedValue);
        }
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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? SeedValue);
        double dropout = o.Stage == GlmVoiceTrainingStage.FineTuning ? o.FineTuningDropout : o.DropoutRate;
        _lm = new GlmDecoder<T>(Engine, _lmLayers, LanguageModelConfiguration(dropout), random);
        _speechTokenizer = new GlmSpeechTokenizer<T>(Engine, _speechTokenizerLayers, SpeechTokenizerConfiguration, random,
            withTrainingStack: o.IncludeSpeechTokenizerTraining);
        _speechDecoder = new GlmVoiceDecoder<T>(Engine, _flowLayers, _hiftLayers, SpeechDecoderConfiguration, random);
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        // The speech tokenizer and decoder, so clones and saved models keep them.
        ComponentLayers.AddRange(_speechTokenizerLayers);
        ComponentLayers.AddRange(_flowLayers);
        ComponentLayers.AddRange(_hiftLayers);
    }

    // ---------------------------------------------------------------- tokens

    private GlmTokenizer Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            if (_options.TokenizerDirectory is null)
                throw new InvalidOperationException("Text needs GLM-4's tokenizer: set GLM4VoiceOptions.TokenizerDirectory or call LoadLanguageModel.");
            _tokenizer = GlmTokenizer.FromDirectory(_options.TokenizerDirectory);
            if (_tokenizer.VocabularySize > _options.VocabSize)
                throw new InvalidOperationException($"The tokenizer has {_tokenizer.VocabularySize} ids; the language model's vocabulary is {_options.VocabSize}.");
            // Speech token c is <|audio_0|>'s id plus c. The release's tokenizer stops at <|audio_16382|>, one short of its
            // codebook; the language model's rows (168960) hold them all.
            int first = _tokenizer.TokenId("<|audio_0|>");
            if (first + _options.SpeechCodebookSize > _options.VocabSize)
                throw new InvalidOperationException($"The speech tokens end at id {first + _options.SpeechCodebookSize - 1}, beyond the language model's {_options.VocabSize}.");
            return _tokenizer;
        }
    }

    private int Special(string token) => Tokenizer.TokenId(token);

    /// <summary>The id of <c>&lt;|audio_0|&gt;</c>: speech token c is this plus c.</summary>
    public int AudioOffset => Special("<|audio_0|>");

    private bool IsSpeech(int id) => id >= AudioOffset && id < AudioOffset + _options.SpeechCodebookSize;

    /// <summary>The text token ids of <paramref name="text"/>, without the <c>[gMASK]&lt;sop&gt;</c> prefix.</summary>
    public int[] TokenizeText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Tokenizer.Encode(text, addPrefix: false);
    }

    /// <summary>The text of token ids; speech tokens are dropped.</summary>
    public string DecodeText(IEnumerable<int> ids) => Tokenizer.Decode(ids.Where(id => !IsSpeech(id)));

    /// <summary>The speech tokens of 16 kHz audio (<c>extract_speech_token</c>: 30-second chunks, the valid frames' codes).</summary>
    public int[] SpeechTokensOf(Tensor<T> recording)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        using var _ = new NoGradScope<T>();
        return SpeechTokenizer.Tokenize(Samples(recording));
    }

    private static double[] Samples(Tensor<T> audio)
    {
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = Ops.ToDouble(audio.GetFlat(i));
        return samples;
    }

    private Tensor<T> To16k(Tensor<T> audio) =>
        AudioHelper<T>.ResampleBandLimited(Engine, Engine.Reshape(audio, new[] { audio.Length }), _options.SampleRate, SpeechSampleRate);

    // ---------------------------------------------------------------- prompts

    // The system prompts of the release's web demo (streaming thoughts) and of Zeng et al. 2024a App. F (text first).
    private const string SpeechInstructionPrompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. ";
    private const string TextInstructionPrompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens.";
    private const string TextFirstPrompt = "User will provide you with a speech instruction. Think about the instruction and speak the response aloud directly.";

    private string SystemPrompt(bool speechInput) => _options.SystemPrompt
        ?? (_options.Generation == GlmVoiceGeneration.TextFirst ? TextFirstPrompt : speechInput ? SpeechInstructionPrompt : TextInstructionPrompt);

    /// <summary>The ids of a speech input: <c>&lt;|begin_of_audio|&gt;</c>, the tokens, <c>&lt;|end_of_audio|&gt;</c>.</summary>
    internal int[] SpeechSpan(IReadOnlyList<int> tokens) =>
        new[] { Special("<|begin_of_audio|>") }.Concat(tokens.Select(c => AudioOffset + c)).Concat(new[] { Special("<|end_of_audio|>") }).ToArray();

    /// <summary>
    /// The prompt up to the reply: <c>[gMASK]&lt;sop&gt;&lt;|system|&gt;\n{system}&lt;|user|&gt;\n{input}&lt;|assistant|&gt;</c>
    /// and <c>streaming_transcription\n</c> (or <c>transcript\n</c> for text-first generation), as the web demo writes it
    /// and the tokenizer splits it at the special tokens.
    /// </summary>
    internal int[] ChatPrompt(string? text, IReadOnlyList<int>? speech) =>
        speech is not null ? PromptAround(TokenizeText("\n").Concat(SpeechSpan(speech)).ToArray(), speechInput: true)
            : PromptAround(TokenizeText("\n" + (text ?? throw new ArgumentNullException(nameof(text)))), speechInput: false);

    // The prompt around the user turn's ids (its leading newline included).
    private int[] PromptAround(IReadOnlyList<int> user, bool speechInput)
    {
        var ids = new List<int>(Tokenizer.Prefix) { Special("<|system|>") };
        ids.AddRange(TokenizeText("\n" + SystemPrompt(speechInput)));
        ids.Add(Special("<|user|>"));
        ids.AddRange(user);
        ids.Add(Special("<|assistant|>"));
        ids.AddRange(TokenizeText(_options.Generation == GlmVoiceGeneration.TextFirst ? "transcript\n" : "streaming_transcription\n"));
        return ids.ToArray();
    }

    /// <summary>The streaming-thoughts reply: 13 text tokens, 26 speech ids, alternating; what remains of either follows.</summary>
    internal List<int> Interleave(IReadOnlyList<int> text, IReadOnlyList<int> speechIds)
    {
        var ids = new List<int>();
        int t = 0, s = 0;
        while (t < text.Count || s < speechIds.Count)
        {
            for (int i = 0; i < _options.TextTokensPerChunk && t < text.Count; i++) ids.Add(text[t++]);
            for (int i = 0; i < _options.SpeechTokensPerChunk && s < speechIds.Count; i++) ids.Add(speechIds[s++]);
        }
        return ids;
    }

    // ---------------------------------------------------------------- generation

    // Each call samples from a generator seeded afresh, so a call's output depends on its input and the seed only.
    private Random SamplingRandom => _samplingRandom ?? throw new InvalidOperationException("Sampling happens inside an inference call.");

    /// <summary>
    /// Transformers' sampling: logits ÷ temperature, then top-p (the smallest set of the most likely tokens whose
    /// probability reaches p, at least one), then a draw; <paramref name="allowed"/> keeps only those ids.
    /// </summary>
    internal static int Sample(double[] logits, double temperature, double topP, Func<int, bool>? allowed, Random random)
    {
        var candidates = new List<(int Id, double Logit)>();
        for (int i = 0; i < logits.Length; i++)
            if (allowed is null || allowed(i)) candidates.Add((i, logits[i] / temperature));
        if (candidates.Count == 0) throw new InvalidOperationException("No token may be sampled.");
        double max = candidates.Max(c => c.Logit);
        var sorted = candidates.Select(c => (c.Id, P: Math.Exp(c.Logit - max))).OrderBy(c => c.P).ToList();
        double total = sorted.Sum(c => c.P);
        // TopPLogitsWarper: sorted ascending, drop the tokens whose cumulative probability is at most 1 − p; keep the last.
        double cumulative = 0;
        var kept = new List<(int Id, double P)>();
        for (int i = 0; i < sorted.Count; i++)
        {
            cumulative += sorted[i].P / total;
            if (cumulative > 1 - topP || i == sorted.Count - 1) kept.Add(sorted[i]);
        }
        double mass = kept.Sum(c => c.P), target = random.NextDouble() * mass;
        foreach (var (id, p) in kept)
        {
            target -= p;
            if (target < 0) return id;
        }
        return kept[kept.Count - 1].Id;
    }

    private sealed class Session
    {
        public Session(GlmDecoder<T> lm) => Cache = lm.NewCache();
        public GlmDecoder<T>.Cache Cache { get; }
        public double[]? Logits { get; set; }
    }

    // Feeds ids, keeping the logits of the last.
    private void Feed(Session session, IReadOnlyList<int> ids)
    {
        if (ids.Count == 0) return;
        if (session.Cache.Length + ids.Count > _options.MaxTextLength)
            throw new InvalidOperationException($"The sequence would exceed {_options.MaxTextLength} tokens.");
        var hidden = LanguageModel.Forward(LanguageModel.Embed(ids), session.Cache);
        var logits = LanguageModel.Logits(Engine.TensorNarrow(hidden, 0, ids.Count - 1, 1));
        var values = new double[logits.Length];
        for (int i = 0; i < values.Length; i++) values[i] = Ops.ToDouble(logits.GetFlat(i));
        session.Logits = values;
    }

    private int Draw(Session session, Func<int, bool>? allowed) =>
        Sample(session.Logits ?? throw new InvalidOperationException("Nothing has been fed."), _options.Temperature, _options.TopP, allowed, SamplingRandom);

    // Samples until one of `stops` (not kept) or `budget` tokens, feeding each back.
    private List<int> Run(Session session, ISet<int> stops, Func<int, bool>? allowed, int budget)
    {
        var generated = new List<int>();
        while (generated.Count < budget)
        {
            int id = Draw(session, allowed);
            if (stops.Contains(id)) break;
            generated.Add(id);
            Feed(session, new[] { id });
        }
        return generated;
    }

    private HashSet<int> EndOfTurn => new() { Special("<|endoftext|>"), Special("<|user|>"), Special("<|observation|>") };

    private TResult Inference<TResult>(Func<TResult> run)
    {
        ThrowIfDisposed();
        bool wasTraining = IsTrainingMode;
        var outer = _samplingRandom;
        SetTrainingMode(false);
        _samplingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
        try
        {
            using var _ = new NoGradScope<T>();
            return run();
        }
        finally
        {
            _samplingRandom = outer;
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>The reply's ids after <paramref name="prompt"/>: free-running streaming thoughts until the end of the turn,
    /// or (text first) the text until <c>&lt;|assistant|&gt;</c>, then the speech until <c>&lt;|end_of_audio|&gt;</c>.</summary>
    internal List<int> GenerateReply(int[] prompt)
    {
        var session = new Session(LanguageModel);
        Feed(session, prompt);
        if (_options.Generation == GlmVoiceGeneration.StreamingThoughts)
            return Run(session, EndOfTurn, null, _options.MaxNewTokens);
        var stops = EndOfTurn;
        stops.Add(Special("<|assistant|>"));
        var reply = Run(session, stops, id => !IsSpeech(id), _options.MaxNewTokens);
        Feed(session, new[] { Special("<|assistant|>") }.Concat(TokenizeText("\n")).Concat(new[] { Special("<|begin_of_audio|>") }).ToArray());
        reply.AddRange(SpeechOnly(session, Math.Max(0, _options.MaxNewTokens - reply.Count)));
        return reply;
    }

    // Speech ids until <|end_of_audio|>.
    private List<int> SpeechOnly(Session session, int budget)
    {
        int end = Special("<|end_of_audio|>");
        return Run(session, new HashSet<int> { end }, id => IsSpeech(id) || id == end, budget);
    }

    /// <summary>
    /// The speech ids for <paramref name="text"/> read aloud (<see cref="GLM4VoiceOptions.TtsFormat"/>): forced-text
    /// thoughts feed the text 13 tokens at a time, sampling up to 26 speech tokens after each and, once the text is in,
    /// speech until the end of the turn; text first feeds the whole text as the transcript; the interleaved base format
    /// continues <c>{text}&lt;|begin_of_audio|&gt;</c>. <see cref="GLM4VoiceOptions.MaxNewTokens"/> caps the speech
    /// tokens.
    /// </summary>
    internal List<int> GenerateSpeechFor(int[] text)
    {
        var session = new Session(LanguageModel);
        var o = _options;
        if (o.TtsFormat == GlmVoiceTtsFormat.InterleavedBase)
        {
            Feed(session, Tokenizer.Prefix.Concat(text).Concat(new[] { Special("<|begin_of_audio|>") }).ToArray());
            return SpeechOnly(session, o.MaxNewTokens);
        }
        // The text is the user's turn (after its newline) and the reply's text.
        Feed(session, PromptAround(TokenizeText("\n").Concat(text).ToArray(), speechInput: false));
        if (o.Generation == GlmVoiceGeneration.TextFirst)
        {
            Feed(session, text.Concat(new[] { Special("<|assistant|>") }).Concat(TokenizeText("\n")).Concat(new[] { Special("<|begin_of_audio|>") }).ToArray());
            return SpeechOnly(session, o.MaxNewTokens);
        }
        var speech = new List<int>();
        var end = EndOfTurn;
        for (int t = 0; t < text.Length && speech.Count < o.MaxNewTokens; t += o.TextTokensPerChunk)
        {
            Feed(session, text.Skip(t).Take(o.TextTokensPerChunk).ToArray());
            bool last = t + o.TextTokensPerChunk >= text.Length;
            int left = o.MaxNewTokens - speech.Count;
            int budget = last ? left : Math.Min(o.SpeechTokensPerChunk, left);
            var chunk = Run(session, last ? end : new HashSet<int>(), id => IsSpeech(id) || (last && end.Contains(id)), budget);
            speech.AddRange(chunk);
        }
        return speech;
    }

    /// <summary>Speech tokens (codes) to a 22.05 kHz waveform with the speech decoder (<see cref="GLM4VoiceOptions.Decoding"/>).</summary>
    public Tensor<T> DecodeSpeech(IReadOnlyList<int> speechTokens)
    {
        if (speechTokens is null) throw new ArgumentNullException(nameof(speechTokens));
        if (speechTokens.Count == 0) return new Tensor<T>(new[] { 0 });
        return Inference(() => SpeechDecoder.Decode(speechTokens, new Tensor<T>(new[] { SpeechDecoder.Flow.Configuration.SpeakerDim }),
            _options.Decoding, SamplingRandom));
    }

    private GlmVoiceReply<T> Reply(List<int> ids)
    {
        var codes = ids.Where(IsSpeech).Select(id => id - AudioOffset).ToArray();
        return new GlmVoiceReply<T> { Text = DecodeText(ids.Where(id => !IsSpeech(id))), SpeechTokens = codes, Audio = DecodeSpeech(codes) };
    }

    /// <summary>GLM-4-Voice's reply to a text (the web demo's text input).</summary>
    public GlmVoiceReply<T> Chat(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Reply(Inference(() => GenerateReply(ChatPrompt(text, null))));
    }

    /// <summary>GLM-4-Voice's reply to 16 kHz speech (the web demo's audio input).</summary>
    public GlmVoiceReply<T> Chat(Tensor<T> recording)
    {
        var tokens = SpeechTokensOf(recording);
        if (tokens.Length == 0) throw new ArgumentException("The recording gives no speech tokens.", nameof(recording));
        return Reply(Inference(() => GenerateReply(ChatPrompt(null, tokens))));
    }

    // ---------------------------------------------------------------- TTS

    /// <inheritdoc />
    /// <remarks>The text's token ids (<see cref="TokenizeText"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    /// <inheritdoc />
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return PredictCore(PreprocessText(text));
    }

    private int[] TokenIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"Token ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids; the result is that text read aloud at 22.05 kHz.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var text = TokenIds(input, nameof(input));
        if (text.Length == 0) throw new ArgumentException("There are no text tokens.", nameof(input));
        var speech = Inference(() => GenerateSpeechFor(text));
        return DecodeSpeech(speech.Where(IsSpeech).Select(id => id - AudioOffset).ToArray());
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- weights

    private static Func<string, int[], double[]> Reader(Dictionary<string, Tensor<float>> tensors, HashSet<string> used, string what) => (name, _) =>
    {
        if (!tensors.TryGetValue(name, out var tensor)) throw new InvalidDataException($"The {what} checkpoint has no tensor '{name}'.");
        used.Add(name);
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
        return values;
    };

    private static void RequireAllUsed(Dictionary<string, Tensor<float>> tensors, HashSet<string> used, string what)
    {
        var left = tensors.Keys.Where(k => !used.Contains(k)).ToList();
        if (left.Count > 0) throw new InvalidDataException($"The {what} checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
    }

    private static Dictionary<string, Tensor<float>> ReadAll(IEnumerable<string> paths)
    {
        var tensors = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        foreach (var path in paths)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
            foreach (var pair in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path)) tensors[pair.Key] = pair.Value;
        }
        return tensors;
    }

    /// <summary>Loads THUDM/glm-4-voice-9b (its safetensors shards) into the language model and, unless one is set, its
    /// tokenizer.</summary>
    public void LoadLanguageModel(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        string index = Path.Combine(directory, "model.safetensors.index.json");
        var files = File.Exists(index)
            ? Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(index))["weight_map"]!.Values<string>().Distinct().Select(f => Path.Combine(directory, f!)).ToList()
            : new List<string> { Path.Combine(directory, "model.safetensors") };
        var tensors = ReadAll(files);
        var used = new HashSet<string>(StringComparer.Ordinal);
        LanguageModel.Load(Reader(tensors, used, "GLM-4-Voice"));
        // The rotary table is recomputed.
        foreach (var key in tensors.Keys.Where(k => k.EndsWith("rotary_pos_emb.inv_freq", StringComparison.Ordinal))) used.Add(key);
        RequireAllUsed(tensors, used, "GLM-4-Voice");
        if (_options.TokenizerDirectory is null && File.Exists(Path.Combine(directory, "tokenizer.model")))
        {
            _options.TokenizerDirectory = directory;
            _tokenizer = null;
        }
    }

    /// <summary>Loads THUDM/glm-4-voice-tokenizer (<c>model.safetensors</c>: the encoder up to the quantizer, the codebook
    /// and its EMA state).</summary>
    public void LoadSpeechTokenizer(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        var tensors = ReadAll(new[] { Path.Combine(directory, "model.safetensors") });
        var used = new HashSet<string>(StringComparer.Ordinal);
        SpeechTokenizer.LoadEncoder(Reader(tensors, used, "speech tokenizer"), tensors.ContainsKey);
        RequireAllUsed(tensors, used, "speech tokenizer");
    }

    /// <summary>Loads THUDM/glm-4-voice-decoder (<c>flow.pt</c> and <c>hift.pt</c>).</summary>
    public void LoadSpeechDecoder(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        var flow = ReadAll(new[] { Path.Combine(directory, "flow.pt") });
        var hift = ReadAll(new[] { Path.Combine(directory, "hift.pt") });
        var flowUsed = new HashSet<string>(StringComparer.Ordinal);
        var hiftUsed = new HashSet<string>(StringComparer.Ordinal);
        SpeechDecoder.Load(Reader(flow, flowUsed, "flow"), Reader(hift, hiftUsed, "HiFT"), hift.ContainsKey);
        RequireAllUsed(flow, flowUsed, "flow");
        RequireAllUsed(hift, hiftUsed, "HiFT");
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "GLM4Voice-Native" : "GLM4Voice-ONNX",
            Description = "GLM-4-Voice: towards intelligent and human-like end-to-end spoken chatbot (Zeng et al., 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "GLM4Voice";
        metadata.AdditionalInfo["Generation"] = _options.Generation.ToString();
        metadata.AdditionalInfo["TtsFormat"] = _options.TtsFormat.ToString();
        metadata.AdditionalInfo["Decoding"] = _options.Decoding.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GLM4Voice<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
