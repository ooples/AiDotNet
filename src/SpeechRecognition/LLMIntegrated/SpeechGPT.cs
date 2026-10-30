using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>
/// SpeechGPT's speech recognition (Zhang et al. 2023): the same LLaMA over text and unit tokens as
/// <see cref="AiDotNet.TextToSpeech.MultiModal.SpeechGPT{T}"/>, given its ASR instruction and a recording's mHuBERT units, writing the transcript.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The recording becomes merged mHuBERT layer-11 units, written <c>&lt;sosp&gt;&lt;u&gt;…&lt;eosp&gt;</c>
/// (§3.1); the question is <c>{ASR description} This is input: {units}</c> in the release's template, and the model
/// continues with the transcript. Training is stage 2's cross-modal instruction fine-tuning, the loss on the
/// transcript (eq. 4), AdamW at 2e-4 with a cosine schedule (App. D and the release's scripts).</para>
/// <para><b>For Beginners:</b> SpeechGPT is a language model that has learned speech units as words. To transcribe, it
/// turns speech into units and writes the text that answers "transcribe this".</para>
/// </remarks>
/// <example>
/// <code>
/// var options = SpeechGPTOptions.Release7B();
/// var model = new SpeechGPT&lt;float&gt;(new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.SpeechRecognition, inputSize: 1, outputSize: 1), options);
/// model.LoadReleasedWeights("SpeechGPT-7B-cm");
/// model.LoadUnitExtractor("mhubert_base_vp_en_es_fr_it3.pt", "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin");
/// var result = model.Transcribe(recording);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities",
    "https://arxiv.org/abs/2305.11000", Year = 2023, Authors = "Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, Xipeng Qiu")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 0.0, MaxGradientNorm = 1.0,
                Schedule = LearningRateSchedulerType.CosineAnnealing, WarmupFraction = 0.03,
                Source = "Zhang et al. 2023 App. D: a peak learning rate of 2e-4; AdamW with a cosine schedule and 3% warmup from "
                        + "the release's scripts.")]
[TaskVariantOf(typeof(AiDotNet.TextToSpeech.MultiModal.SpeechGPT<>), "speech recognition")]
public partial class SpeechGPT<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    private readonly SpeechGPTOptions _options;
    private readonly SpeechGptCore<T> _core;
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _hubertLayers = new();
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;

    /// <summary>Creates SpeechGPT's speech recognizer.</summary>
    public SpeechGPT(NeuralNetworkArchitecture<T> architecture, SpeechGPTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SpeechGPTOptions();
        Options = _options;
        base.SampleRate = _options.SampleRate;
        _core = new SpeechGptCore<T>(Engine, _options, _lmLayers, _hubertLayers, new List<LayerBase<T>>(), new List<LayerBase<T>>(),
            architecture.RandomSeed ?? _options.Seed ?? 42, withVocoder: false);
        Layers.AddRange(_lmLayers);
        // The unit extractor, so clones and saved models keep it; it does not train.
        Layers.AddRange(_hubertLayers);
    }

    internal SpeechGptCore<T> Core => _core;

    /// <inheritdoc />
    /// <remarks>SpeechInstruct's ASR data is English (§3.1).</remarks>
    public IReadOnlyList<string> SupportedLanguages => new[] { "en" };

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    /// <summary>The token ids of <paramref name="text"/> (no <c>&lt;s&gt;</c>).</summary>
    public int[] TokenizeText(string text) => _core.Tokenizer.Encode(text, addBos: false);

    // ---------------------------------------------------------------- weights

    /// <summary>Loads fnlp/SpeechGPT-7B-cm (use <see cref="SpeechGPTOptions.Release7B"/>) and, unless one is set, its tokenizer.</summary>
    public void LoadReleasedWeights(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        string index = Path.Combine(directory, "pytorch_model.bin.index.json");
        if (!File.Exists(index)) index = Path.Combine(directory, "model.safetensors.index.json");
        var files = File.Exists(index)
            ? Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(index))["weight_map"]!.Values<string>().Distinct().Select(f => Path.Combine(directory, f!)).ToList()
            : new List<string> { File.Exists(Path.Combine(directory, "model.safetensors")) ? Path.Combine(directory, "model.safetensors") : Path.Combine(directory, "pytorch_model.bin") };
        var tensors = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        foreach (var file in files)
            foreach (var pair in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file)) tensors[pair.Key] = pair.Value;
        _core.LanguageModel.LoadTorchWeights((name, _) =>
        {
            var t = tensors.TryGetValue(name, out var tensor) ? tensor : throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var values = new double[t.Length];
            for (int i = 0; i < values.Length; i++) values[i] = t.GetFlat(i);
            return values;
        }, "model.", "lm_head.weight");
        if (_options.TokenizerDirectory is null && File.Exists(Path.Combine(directory, "tokenizer.model")))
        {
            _options.TokenizerDirectory = directory;
            _core.ResetTokenizer();
        }
    }

    /// <summary>Loads mHuBERT and its k-means model, the unit extractor.</summary>
    public void LoadUnitExtractor(string hubertCheckpoint, string kmeansModel)
    {
        ThrowIfDisposed();
        var tensors = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(hubertCheckpoint);
        _core.Units.LoadFairseqWeights(name =>
        {
            if (!tensors.TryGetValue("model." + name, out var t)) return null;
            var values = new double[t.Length];
            for (int i = 0; i < values.Length; i++) values[i] = t.GetFlat(i);
            return values;
        });
        _core.Units.Centroids = AiDotNet.Audio.Foundations.JoblibKMeans.ReadClusterCenters(kmeansModel);
    }

    // ---------------------------------------------------------------- recognition

    private int[] Prompt(Tensor<T> audio)
    {
        var units = _core.UnitsOf(audio.Rank == 1 ? audio : Engine.Reshape(audio, new[] { audio.Length }));
        if (units.Length == 0) throw new ArgumentException("The recording gives no units.", nameof(audio));
        return _core.Prompt($"{_options.AsrInstruction} This is input: {SpeechGptCore<T>.UnitString(units)}");
    }

    private List<int> Recognize(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return _core.Generate(Prompt(audio), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    /// <remarks>The transcript SpeechGPT writes after its ASR instruction and the recording's units.</remarks>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        var ids = Recognize(audio);
        string text = _core.TextOf(ids);
        double duration = (double)audio.Length / SampleRate;
        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? "en",
            Confidence = NumOps.One,
            DurationSeconds = duration,
            Segments = ResolveReturnTimestamps(includeTimestamps) && text.Length > 0
                ? new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.One } }
                : Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null,
        CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc />
    /// <remarks>SpeechGPT has no language-identification task; its ASR data is English.</remarks>
    public string DetectLanguage(Tensor<T> audio) =>
        throw new NotSupportedException("SpeechGPT has no language-identification task (§3.1: its ASR data is English).");

    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) =>
        throw new NotSupportedException("SpeechGPT has no language-identification task (§3.1: its ASR data is English).");

    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) =>
        throw new NotSupportedException("SpeechGPT transcribes a whole recording.");

    // ---------------------------------------------------------------- the network

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
    }

    /// <inheritdoc />
    /// <remarks>The language model's layers are not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a 16 kHz recording; the result is the transcript's token ids.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var ids = Recognize(input);
        var text = ids.Where(id => !_core.IsUnit(id)).ToArray();
        return IndexTtsOps.Ids<T>(text.Length == 0 ? new[] { _core.Tokenizer.EosId } : text);
    }

    /// <inheritdoc />
    /// <remarks>Transcripts are token ids.</remarks>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => LayerInputDomain.Indices(_options.VocabSize);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <summary>
    /// One update on a 16 kHz recording <paramref name="input"/> and its transcript's token ids
    /// <paramref name="expected"/> (<see cref="TokenizeText"/>): the ASR instruction with the loss on the transcript, its
    /// <c>&lt;eos&gt;</c> and the end (eq. 4).
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        ThrowIfDisposed();
        var (ids, first) = Example(input, expected);
        TrainWithCustomObjective(input, expected, (_, _) => _core.Loss(ids, first), Optimizer());
        return true;
    }

    /// <inheritdoc />
    public TrainingObjectiveKind TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <inheritdoc />
    public Tensor<T> ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The cross-entropy <see cref="TryApplyLearningRule"/> lowers: the transcript after the ASR instruction.</remarks>
    public T EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        var (ids, first) = Example(input, target);
        using var _ = new NoGradScope<T>();
        return _core.Loss(ids, first).GetFlat(0);
    }

    // The ASR instruction on the recording's units and the transcript's ids, and where the loss starts.
    private (int[] Ids, int First) Example(Tensor<T> input, Tensor<T> expected)
    {
        var transcript = new int[expected.Length];
        for (int i = 0; i < transcript.Length; i++)
        {
            transcript[i] = (int)Math.Round(NumOps.ToDouble(expected.GetFlat(i)));
            if (transcript[i] < 0 || transcript[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(expected), $"Transcript ids lie in [0, {_options.VocabSize}).");
        }
        var units = _core.UnitsOf(input.Rank == 1 ? input : Engine.Reshape(input, new[] { input.Length }));
        string prefix = _options.MetaInstruction + $"[Human]: {_options.AsrInstruction} This is input: {SpeechGptCore<T>.UnitString(units)} <eoh>. [SpeechGPT]: ";
        return _core.Example(prefix, _core.Tokenizer.Decode(transcript) + "<eos>");
    }

    /// <inheritdoc />
    /// <remarks>LLaMA trains; mHuBERT is frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_lmLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        int total = Math.Max(1, o.TotalSteps > 0 ? o.TotalSteps : 4000), warmup = (int)Math.Ceiling(o.WarmupFraction * total);
        return _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
            LearningRateScheduler = new LinearWarmupScheduler(o.LearningRate, warmup, total, 0.0, LinearWarmupScheduler.DecayMode.Cosine, 0.0),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        }));
    }

    /// <inheritdoc />
    /// <remarks>The unit extractor's k-means centroids are fitted, not trained.</remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        _core.Units.DeclareCentroids(state, "UnitCentroids");
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => true;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = "SpeechGPT-ASR",
        Description = "SpeechGPT speech recognition (Zhang et al., 2023)",
        FeatureCount = _options.HiddenDim,
        AdditionalInfo = BaseAudioMetadataInfo(),
    };

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SpeechGPT<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
