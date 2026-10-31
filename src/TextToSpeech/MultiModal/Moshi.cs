using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// Moshi (Défossez et al. 2024): a speech-text foundation model for full-duplex dialogue. Every 80 ms it reads a frame of
/// the user's speech and writes a text token of its inner monologue and a frame of its own speech in Mimi's codes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The paper's model (the defaults): the RQ-Transformer (§3.4) — a 7B Temporal Transformer over the sum of the
/// streams' embeddings, a Depth Transformer predicting the text and then each codebook of each speaker in turn (Eq. 6) —
/// over Mimi (§3.3), a streaming codec with a split RVQ whose first level is distilled from WavLM, and Helium's
/// tokenizer. <see cref="Synthesize"/> is App. C's streaming TTS; <see cref="Chat"/> is the dialogue;
/// <see cref="AiDotNet.SpeechRecognition.Streaming.Moshi{T}"/> is App. C's streaming ASR. Training covers Mimi's
/// adversarial-only objective with distillation (<see cref="TrainMimi"/>) and the language model's four stages
/// (<see cref="TrainConversation"/>, <see cref="TrainText"/>). <see cref="MoshiOptions.Release"/> is kyutai/moshiko, whose
/// weights <see cref="LoadReleasedWeights"/> reads.</para>
/// <para><b>For Beginners:</b> Moshi talks with you in real time: it listens and speaks at once, can be interrupted, and
/// writes down what it is about to say a moment before saying it, which keeps its speech coherent.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
///     inputHeight: 1, inputWidth: 1, inputDepth: 1, outputSize: 1);
/// var moshi = new Moshi&lt;float&gt;(architecture, MoshiOptions.Release());
/// moshi.LoadReleasedWeights("kyutai/moshiko-pytorch-bf16");
/// var (reply, monologue) = moshi.Chat(userSpeech24kHz);
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
    "Moshi: a speech-text foundation model for real-time dialogue",
    "https://arxiv.org/abs/2410.00037",
    Year = 2024,
    Authors = "Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, Neil Zeghidour"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-5, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.1, MaxGradientNorm = 1.0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Défossez et al. 2024 §4.4 and Table 1: AdamW (0.9, 0.95) with weight decay 0.1; pre-training at 3e-5 for the "
                        + "Temporal Transformer and 2e-4 for the Depth Transformer, cosine with linear warmup, the text embedding and "
                        + "linear at 0.75 on audio batches; post-training 3e-6/5e-5, Fisher 2e-6/4e-6, instruct 2e-6, fixed.")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "mimi", LearningRate = 8e-4, Beta1 = 0.5, Beta2 = 0.9, WeightDecay = 5e-2,
                Source = "Défossez et al. 2024 §3.3: Mimi trains with AdamW at 8e-4, betas (0.5, 0.9), weight decay 5e-2 on the "
                        + "Transformers only, and an exponential moving average of the weights at 0.99.")]
public partial class Moshi<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly MoshiOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private MoshiCore<T>? _core;
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _mimiLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private readonly List<LayerBase<T>> _distillationLayers = new();
    private readonly List<LayerBase<T>> _wavLmLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates Moshi running an ONNX export.</summary>
    public Moshi(NeuralNetworkArchitecture<T> architecture, string modelPath, MoshiOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MoshiOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates Moshi with its networks (randomly initialized as the release initializes them).</summary>
    public Moshi(NeuralNetworkArchitecture<T> architecture, MoshiOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MoshiOptions();
        Options = _options;
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    /// <summary>The longest text the TTS takes, in tokens.</summary>
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.Codebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;

    /// <inheritdoc />
    /// <remarks>Mimi's frame rate rounded (12.5 Hz reads as 13).</remarks>
    public int CodecFrameRate => (int)Math.Round((double)_options.SampleRate / MoshiCore<T>.MimiConfiguration(_options).SamplesPerFrame);

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.Codebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.CodebookSize;

    private bool HasPaperLayers => _core is not null;
    private int SeedValue => _options.Seed ?? 42;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal MoshiCore<T> Core => _core ?? throw NotNative();

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
        _core = new MoshiCore<T>(Engine, _options, _lmLayers, _mimiLayers, _discriminatorLayers, _distillationLayers, _wavLmLayers,
            Architecture.RandomSeed ?? SeedValue);
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        // Mimi, and what training it adds, so clones and saved models keep them.
        ComponentLayers.AddRange(_mimiLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
        ComponentLayers.AddRange(_distillationLayers);
        ComponentLayers.AddRange(_wavLmLayers);
    }

    // ---------------------------------------------------------------- text

    /// <summary>Helium's token ids of text (no &lt;s&gt;).</summary>
    public int[] TokenizeText(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Core.Tokenizer.Encode(text);
    }

    // ---------------------------------------------------------------- speech

    /// <summary>
    /// App. C's streaming TTS of <paramref name="text"/> (words split on white space): returns the 24 kHz speech, which
    /// starts <see cref="MoshiOptions.TextAudioDelaySeconds"/> of look-ahead after the first word, and when each word was
    /// spoken. Meant for <see cref="MoshiVariant.Tts"/> weights.
    /// </summary>
    public (Tensor<T> Audio, IReadOnlyList<MoshiTimedWord> Words) Speak(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        using var _ = Inference();
        var (audio, words) = Core.Speak(text, SamplingRandom());
        return (audio, words);
    }

    /// <summary>
    /// The dialogue: Moshi listens to the user's 24 kHz <paramref name="userSpeech"/> frame by frame and answers as it goes.
    /// Returns its speech over the same span and the text of its inner monologue.
    /// </summary>
    public (Tensor<T> Audio, string Text) Chat(Tensor<T> userSpeech)
    {
        ThrowIfDisposed();
        if (userSpeech is null) throw new ArgumentNullException(nameof(userSpeech));
        using var _ = Inference();
        return Core.Chat(userSpeech, SamplingRandom());
    }

    /// <inheritdoc />
    /// <remarks>App. C's streaming TTS (<see cref="Speak"/>).</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return Speak(text).Audio;
    }

    /// <inheritdoc />
    /// <remarks>Helium's token ids of the text.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is Helium token ids; the result is the speech App. C's TTS makes of their text.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var ids = new int[input.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(input.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(nameof(input), $"Token ids lie in [0, {_options.VocabSize}).");
        }
        string text = Core.TextOf(ids);
        return string.IsNullOrWhiteSpace(text) ? new Tensor<T>(new[] { 0 }) : Speak(text).Audio;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- the codec

    /// <inheritdoc />
    /// <remarks>Mimi's first <see cref="MoshiOptions.Codebooks"/> codes of a 24 kHz recording, <c>[frames, Q]</c>.</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (!HasPaperLayers) throw NotNative();
        using var _ = Inference();
        var codes = Core.AudioCodes(audio);
        var tokens = new Tensor<T>(new[] { codes.GetLength(1), codes.GetLength(0) });
        for (int t = 0; t < codes.GetLength(1); t++)
            for (int k = 0; k < codes.GetLength(0); k++) tokens[t, k] = Ops.FromDouble(codes[k, t]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>Mimi's audio of codes <c>[frames, Q]</c>.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (!HasPaperLayers) throw NotNative();
        int q = _options.Codebooks;
        if (tokens.Rank != 2 || tokens.Shape[1] != q) throw new ArgumentException($"Moshi's codes are [frames, {q}].", nameof(tokens));
        var codes = new int[q, tokens.Shape[0]];
        for (int t = 0; t < tokens.Shape[0]; t++)
            for (int k = 0; k < q; k++)
            {
                codes[k, t] = (int)Math.Round(Ops.ToDouble(tokens[t, k]));
                if (codes[k, t] < 0 || codes[k, t] >= _options.CodebookSize) throw new ArgumentException($"Codes lie in [0, {_options.CodebookSize}).", nameof(tokens));
            }
        using var _ = Inference();
        return Core.Audio(codes);
    }

    // Evaluation mode without gradients, restored afterwards.
    private IDisposable Inference()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        var scope = new NoGradScope<T>();
        return new Restore(() =>
        {
            scope.Dispose();
            SetTrainingMode(wasTraining);
        });
    }

    private sealed class Restore : IDisposable
    {
        private Action? _action;
        public Restore(Action action) => _action = action;
        public void Dispose()
        {
            _action?.Invoke();
            _action = null;
        }
    }

    // A fresh generator per call: the same input gives the same speech.
    private Random SamplingRandom() => AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Moshi-Native" : "Moshi-ONNX",
            Description = "Moshi: a speech-text foundation model for real-time dialogue (Défossez et al., 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "Moshi";
        metadata.AdditionalInfo["Variant"] = _options.Variant.ToString();
        metadata.AdditionalInfo["DepthLayout"] = _options.DepthLayout.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Moshi<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
