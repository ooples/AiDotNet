using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.SpeechRecognition.Streaming;

/// <summary>
/// Moshi's streaming speech recognizer (Défossez et al. 2024 App. C): the same RQ-Transformer and Mimi as
/// <see cref="AiDotNet.TextToSpeech.MultiModal.Moshi{T}"/>, trained with the text stream 2 s behind the audio, so that at
/// inference the recording's codes are forced into the audio streams and the text is decoded greedily, with each word's
/// time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The options default to <see cref="MoshiVariant.Asr"/>. Training pairs a recording with its transcript's Helium
/// ids; without word times, every word is taken to start at the beginning, so the transcript runs from the first frame
/// after the delay (the release interleaver's keep-and-shift for zero timestamps). <see cref="TrainConversation"/> takes
/// word times, as the paper's Whisper alignment gives them.</para>
/// <para><b>For Beginners:</b> This listens to speech and writes it down as it goes, about two seconds behind.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Moshi: a speech-text foundation model for real-time dialogue", "https://arxiv.org/abs/2410.00037", Year = 2024,
    Authors = "Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, Neil Zeghidour")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-5, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.1, MaxGradientNorm = 1.0,
                Source = "Défossez et al. 2024 §4.4 and Table 1: AdamW (0.9, 0.95) with weight decay 0.1 at the stage's rates.")]
[TaskVariantOf(typeof(AiDotNet.TextToSpeech.MultiModal.Moshi<>), "speech recognition")]
public partial class Moshi<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    private readonly MoshiOptions _options;
    private readonly MoshiCore<T> _core;
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _mimiLayers = new();
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;

    /// <summary>Creates the recognizer (randomly initialized as the release initializes it).</summary>
    public Moshi(NeuralNetworkArchitecture<T> architecture, MoshiOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MoshiOptions { Variant = MoshiVariant.Asr };
        Options = _options;
        base.SampleRate = _options.SampleRate;
        _core = new MoshiCore<T>(Engine, _options, _lmLayers, _mimiLayers, new List<LayerBase<T>>(), new List<LayerBase<T>>(),
            new List<LayerBase<T>>(), architecture.RandomSeed ?? _options.Seed ?? 42);
        Layers.AddRange(_lmLayers);
        // Mimi, so clones and saved models keep it; the recognizer does not train it.
        Layers.AddRange(_mimiLayers);
    }

    internal MoshiCore<T> Core => _core;

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages => new[] { "en" };

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => true;

    /// <summary>Helium's token ids of text (no &lt;s&gt;).</summary>
    public int[] TokenizeText(string text) => _core.Tokenizer.Encode(text ?? throw new ArgumentNullException(nameof(text)));

    /// <summary>Loads released weights: <c>model.safetensors</c> (the language model), Mimi's <c>tokenizer-*.safetensors</c>
    /// and Helium's tokenizer.</summary>
    public void LoadReleasedWeights(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        Dictionary<string, double[]> Read(string path)
        {
            var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
                state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
            return state;
        }
        var lm = Read(Path.Combine(directory, "model.safetensors"));
        _core.LanguageModel.LoadTorchWeights(name => lm.TryGetValue(name, out var v) ? v : null);
        string? mimi = Directory.GetFiles(directory, "tokenizer-*.safetensors").OrderBy(f => f, StringComparer.Ordinal).FirstOrDefault();
        if (mimi is not null)
        {
            var state = Read(mimi);
            _core.Mimi.Load(name => state.TryGetValue(name, out var v) ? v : null);
        }
        string tokenizer = Path.Combine(directory, "tokenizer_spm_32k_3.model");
        if (_options.TokenizerPath is null && File.Exists(tokenizer))
        {
            _options.TokenizerPath = tokenizer;
            _core.ResetTokenizer();
        }
    }

    // ---------------------------------------------------------------- recognition

    private (string Text, List<MoshiTimedWord> Words) Recognize(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return _core.Transcribe(audio.Rank == 1 ? audio : Engine.Reshape(audio, new[] { audio.Length }));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        var (text, words) = Recognize(audio);
        double duration = (double)audio.Length / SampleRate;
        var segments = new List<TranscriptionSegment<T>>();
        if (ResolveReturnTimestamps(includeTimestamps))
            for (int i = 0; i < words.Count; i++)
                segments.Add(new TranscriptionSegment<T>
                {
                    Text = words[i].Text.Trim(),
                    StartTime = words[i].Time,
                    EndTime = i + 1 < words.Count ? Math.Max(words[i].Time, words[i + 1].Time) : Math.Max(words[i].Time, duration),
                    Confidence = NumOps.One,
                });
        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? "en",
            Confidence = NumOps.One,
            DurationSeconds = duration,
            Segments = segments,
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null,
        CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc />
    public string DetectLanguage(Tensor<T> audio) =>
        throw new NotSupportedException("Moshi has no language-identification task (its speech data is English).");

    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) =>
        throw new NotSupportedException("Moshi has no language-identification task (its speech data is English).");

    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) =>
        throw new NotSupportedException("This recognizer transcribes a whole recording; its decoding is frame-synchronous with a fixed 2 s delay.");

    // ---------------------------------------------------------------- the network

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    /// <remarks>The transcript's Helium ids of a 24 kHz recording.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var (text, _) = Recognize(input);
        var ids = _core.Tokenizer.Encode(text);
        return IndexTtsOps.Ids<T>(ids.Length == 0 ? new[] { _core.Tokenizer.EosId } : ids);
    }

    /// <inheritdoc />
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => LayerInputDomain.Indices(_options.VocabSize);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    // ---------------------------------------------------------------- training

    // The joint sequence of a recording and its transcript ids, every word taken to start at the beginning.
    private int[,] Example(Tensor<T> input, Tensor<T> expected)
    {
        var transcript = new int[expected.Length];
        for (int i = 0; i < transcript.Length; i++)
        {
            transcript[i] = (int)Math.Round(NumOps.ToDouble(expected.GetFlat(i)));
            if (transcript[i] < 0 || transcript[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(expected), $"Transcript ids lie in [0, {_options.VocabSize}).");
        }
        int[,] codes;
        using (new NoGradScope<T>()) codes = _core.AudioCodes(input.Rank == 1 ? input : Engine.Reshape(input, new[] { input.Length }));
        int frames = codes.GetLength(1);
        double end = frames / _core.FrameRate;
        var text = _core.TextStream(new[] { new MoshiTokenSpan(transcript, 0, end) }, frames, AudioDelay());
        return _core.Sequence(text, codes, null);
    }

    private double AudioDelay() => _options.Variant switch
    {
        MoshiVariant.Asr => -_options.TextAudioDelaySeconds,
        MoshiVariant.Tts => _options.TextAudioDelaySeconds,
        _ => 0.0,
    };

    /// <summary>One step on a recording with its words and times (the main stream; <see cref="MoshiConversation{T}.User"/>
    /// is not read). Returns the loss.</summary>
    public T TrainConversation(MoshiConversation<T> conversation)
    {
        ThrowIfDisposed();
        if (conversation is null) throw new ArgumentNullException(nameof(conversation));
        int[,] codes;
        using (new NoGradScope<T>()) codes = _core.AudioCodes(conversation.Moshi);
        int frames = codes.GetLength(1);
        var text = conversation.Words is null ? Enumerable.Repeat(-1, frames).ToArray() : _core.TextStream(conversation.Words, frames, AudioDelay());
        var sequence = _core.Sequence(text, codes, null);
        var marker = IndexTtsOps.Ids<T>(new[] { 0 });
        return TrainWithCustomObjective(marker, marker, (_, _) => _core.Loss(sequence), Optimizer());
    }

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        ThrowIfDisposed();
        var codes = Example(input, expected);
        TrainWithCustomObjective(input, expected, (_, _) => _core.Loss(codes), Optimizer());
        return true;
    }

    /// <inheritdoc />
    public TrainingObjectiveKind TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <inheritdoc />
    public Tensor<T> ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    public T EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        var codes = Example(input, target);
        using var _ = new NoGradScope<T>();
        return _core.Loss(codes).GetFlat(0);
    }

    /// <inheritdoc />
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_lmLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer() =>
        _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, _core.CreateLanguageModelOptimizer(this, _lmLayers, textOnly: false));

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    /// <remarks>Mimi's codebooks (moving averages, not parameters) are model state.</remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        int q = 0;
        double epsilon = _core.Mimi.Configuration.CodebookEpsilon;
        foreach (var book in _core.Mimi.AllCodebooks)
        {
            var b = book;
            state.DeclareDoubleVector($"MimiCodebook{q}.Sum", () => new Vector<double>(b.Sum),
                v => { if (v is not null) b.Load(v.ToArray(), b.Usage, epsilon, b.Initialized); });
            state.DeclareDoubleVector($"MimiCodebook{q}.Usage", () => new Vector<double>(b.Usage),
                v => { if (v is not null) b.Load(b.Sum, v.ToArray(), epsilon, b.Initialized); });
            q++;
        }
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => true;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = "Moshi-ASR",
        Description = "Moshi streaming speech recognition (Défossez et al., 2024, App. C)",
        FeatureCount = _options.HiddenDim,
        AdditionalInfo = BaseAudioMetadataInfo(),
    };

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
