using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>
/// VoxtLM: one decoder-only language model for speech recognition, synthesis, and speech and text continuation
/// (Maiti et al. 2024), here as a speech recognizer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> VoxtLM treats speech as a second language. A speech tokenizer turns the recording into a
/// sequence of discrete "speech words" (clusters of HuBERT features), and one language model, trained on speech, text
/// and both together, reads <c>&lt;startspeech&gt;</c>, the speech words and <c>&lt;generatetext&gt;</c>, then writes
/// the transcript.</para>
/// <para>
/// The defaults are VoxtLM-k50 small as ESPnet's <c>egs2/voxtlm_v1</c> recipe trains it (§3–4 and the recipe's
/// <c>train.yaml</c>): the speech tokenizer is k-means with 50 clusters over HuBERT Base's sixth-layer features
/// (<c>kmeans_feature: hubert_base/6</c>), the text has 2,000 subword tokens, and the language model is ESPnet's
/// <c>TransformerLM</c> of 12 pre-norm layers at width 768 (12 heads, ReLU feed-forward 3,072, no positional encoding,
/// dropout 0.1) with an untied output layer. Training predicts the transcript and <c>&lt;sos/eos&gt;</c> after the
/// condition (Table 1), with label smoothing 0.1, under Adam (1e-3, weight decay 1e-6) with the <c>warmuplr</c> schedule
/// over 10,000 steps and gradient clipping at 5. The speech tokenizer does not train: load HuBERT and a fitted k-means
/// (<see cref="LoadSpeechTokenizer"/>) or fit the k-means here (<see cref="FitSpeechUnits"/>). The paper decodes with
/// beam search (beam 5 in the recipe); <see cref="Transcribe"/> decodes greedily. The recipe also applies BPE over the
/// unit sequences; here each unit is one token.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 2057);
/// var model = new VoxtLM&lt;float&gt;(architecture);
/// model.LoadSpeechTokenizer("hubert_base_ls960.pt", "km_50.mdl");
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("VoxtLM: Unified Decoder-Only Models for Consolidating Speech Recognition/Synthesis and Speech/Text Continuation", "https://arxiv.org/abs/2309.07937", Year = 2024, Authors = "Maiti et al.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, WeightDecay = 1e-6, Schedule = LearningRateSchedulerType.Noam,
                WarmupSteps = 10000, MaxGradientNorm = 5,
                Source = "Maiti et al. 2024, Sec. 4: Adam with a warmup learning rate schedule; ESPnet's egs2/voxtlm_v1 train.yaml: "
                        + "adam lr 0.001, weight_decay 1e-6, warmuplr with 10000 warmup steps, grad_clip 5.")]
public partial class VoxtLM<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The joint vocabulary: the special tokens, the speech units, the text tokens and <c>&lt;sos/eos&gt;</c>.</remarks>
    protected override int OutputFeatureWidth => _network?.VocabSize ?? 1;

    private readonly VoxtLMOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private VoxtLMNetwork<T>? _network;
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _hubertLayers = new();
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal VoxtLMNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public VoxtLM(NeuralNetworkArchitecture<T> architecture, string modelPath, VoxtLMOptions? options = null) : base(architecture)
    {
        _options = options ?? new VoxtLMOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public VoxtLM(NeuralNetworkArchitecture<T> architecture, VoxtLMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new VoxtLMOptions(); Options = _options; _useNativeMode = true;
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer ?? RecipeOptimizer());
    }

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    // Adam under ESPnet's warmuplr, lr · warmup^0.5 · min(step^-0.5, step · warmup^-1.5): the Noam schedule whose factor
    // puts its peak at the learning rate.
    private AdamOptimizer<T, Tensor<T>, Tensor<T>> RecipeOptimizer()
    {
        int warmup = Math.Max(1, _options.WarmupSteps), width = _options.ModelDim;
        var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            EnableGradientClipping = _options.MaxGradientNorm > 0, MaxGradientNorm = _options.MaxGradientNorm > 0 ? _options.MaxGradientNorm : 1.0,
            LearningRateScheduler = new NoamSchedule(width, warmup, _options.LearningRate * Math.Sqrt((double)width * warmup)),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        };
        if (_options.WeightDecay > 0)
            options.Regularization = new AiDotNet.Regularization.L2Regularization<T, Tensor<T>, Tensor<T>>(
                new Models.RegularizationOptions { Strength = _options.WeightDecay });
        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
    }

    /// <summary>
    /// Loads the speech tokenizer: a fairseq HuBERT checkpoint (HuBERT Base for the recipe's <c>hubert_base/6</c>) and a
    /// joblib-saved scikit-learn k-means model fitted on its features.
    /// </summary>
    public void LoadSpeechTokenizer(string hubertCheckpoint, string kmeansModel)
    {
        ThrowIfDisposed();
        var network = _network ?? throw new NotSupportedException("The speech tokenizer belongs to the native network.");
        // safetensors and .pt/.pth archives open as a checkpoint; a torch.save archive under another name is unpickled whole.
        string extension = Path.GetExtension(hubertCheckpoint).ToLowerInvariant();
        using (var file = extension is ".safetensors" or ".pt" or ".pth" or ".tar" or ".ckpt" ? AiDotNet.ModelLoading.TorchCheckpointFile.Open(hubertCheckpoint) : null)
        {
            var checkpoint = file ?? AiDotNet.ModelLoading.TorchTensorSource.From(new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(hubertCheckpoint),
                $"the checkpoint {hubertCheckpoint}");
            network.Units.LoadFairseqWeights(checkpoint.WithPrefix("model."));
        }
        var centroids = AiDotNet.Audio.Foundations.JoblibKMeans.ReadClusterCenters(kmeansModel);
        if (centroids.GetLength(0) != _options.NumSpeechUnits || centroids.GetLength(1) != _options.HubertWidth)
            throw new InvalidDataException($"The k-means model is {centroids.GetLength(0)} x {centroids.GetLength(1)}; the tokenizer has {_options.NumSpeechUnits} x {_options.HubertWidth}.");
        network.Units.Centroids = centroids;
    }

    /// <summary>Fits the speech tokenizer's k-means on the HuBERT features of 16 kHz <paramref name="recordings"/>.</summary>
    public void FitSpeechUnits(IEnumerable<Tensor<T>> recordings, int iterations = 20)
    {
        ThrowIfDisposed();
        var network = _network ?? throw new NotSupportedException("The speech tokenizer belongs to the native network.");
        network.FitUnits(recordings.Select(r => (IReadOnlyList<double>)Samples(r, 0, Math.Min(r.Length, SampleRate * _options.MaxAudioLengthSeconds))),
            iterations, Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
    }

    /// <summary>Transcribes audio: the language model's greedy transcript after the ASR condition.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for VoxtLM.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        int[] tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            tokens = network.Greedy(network.AsrPrompt(Samples(audio, 0, Math.Min(audio.Length, SampleRate * _options.MaxAudioLengthSeconds))), _options.MaxTextLength);
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("VoxtLM transcribes a whole recording.");

    private string Text(int[] tokens)
    {
        var vocabulary = _options.Vocabulary;
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (vocabulary is not null && t < vocabulary.Length) sb.Append(vocabulary[t]);
            else if (vocabulary is null && t <= 0x10FFFF && (t < 0xD800 || t > 0xDFFF)) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The speech tokenizer and language model (see the class remarks). It used to be a generic encoder-LLM template.
        _lmLayers.Clear();
        _hubertLayers.Clear();
        _network = new VoxtLMNetwork<T>(Engine, _options, _lmLayers, _hubertLayers, Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(_lmLayers);
        // The speech tokenizer, so clones and saved models keep it; it does not train.
        Layers.AddRange(_hubertLayers);
    }

    /// <inheritdoc />
    /// <remarks>The language model and the speech tokenizer are not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_network is null) base.ResolveLazyLayerShapes();
    }

    private double[] Samples(Tensor<T> waveform, int offset, int count)
    {
        var samples = new double[count];
        var span = waveform.Data.Span;
        for (int i = 0; i < count; i++) samples[i] = NumOps.ToDouble(span[offset + i]);
        return samples;
    }

    private double[] Recording(Tensor<T> input, int row, int rows)
    {
        int samples = input.Length / rows;
        return Samples(input, row * samples, Math.Min(samples, SampleRate * _options.MaxAudioLengthSeconds));
    }

    // The logits [1, vocabulary] of the transcript's first token for a waveform [S], or [batch, 1, vocabulary] for [batch, S].
    private Tensor<T> FirstLogits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++) outputs[r] = _network.FirstLogits(_network.AsrPrompt(Recording(input, r, rows)));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return FirstLogits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : FirstLogits(input);

    // The transcript a training target names: text-token indices [length], or one distribution over the joint vocabulary
    // per token [length, vocabulary], read as its most likely text tokens.
    private int[] Transcript(VoxtLMNetwork<T> network, Tensor<T> target)
    {
        int vocabulary = network.VocabSize;
        var span = target.Data.Span;
        if (target.Rank >= 2 && target.Shape[^1] == vocabulary)
        {
            var tokens = new int[target.Length / vocabulary];
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = network.TextOffset;
                for (int v = network.TextOffset; v < network.EndToken; v++)
                    if (NumOps.GreaterThan(span[r * vocabulary + v], span[r * vocabulary + best])) best = v;
                tokens[r] = best - network.TextOffset;
            }
            return tokens;
        }
        var ids = new int[target.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value < 0 || value >= _options.TextVocabSize) throw new ArgumentException($"Text token {value} is outside [0, {_options.TextVocabSize}).", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    private Tensor<T> TargetRow(Tensor<T> target, int row, int rows)
    {
        if (rows == 1) return target;
        int perRow = target.Length / rows;
        return Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { row, 0 }, new[] { 1, perRow }),
            target.Rank >= 3 ? target.Shape.ToArray().Skip(1).ToArray() : new[] { perRow });
    }

    private Tensor<T> BatchLoss(VoxtLMNetwork<T> network, Tensor<T> input, Tensor<T> target, bool training)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var loss = network.AsrLoss(network.AsrPrompt(Recording(input, r, rows)), Transcript(network, TargetRow(target, r, rows)), training, _random);
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The label-smoothed loss of the transcript and <c>&lt;sos/eos&gt;</c> after the ASR condition, with dropout.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        TrainWithCustomLoss(input, _ => BatchLoss(network, input, target, training: true));
        return true;
    }

    /// <inheritdoc />
    /// <remarks>The language model trains; the speech tokenizer (HuBERT and its k-means) is fixed.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_network is null) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_lmLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    /// <inheritdoc />
    /// <remarks>The speech tokenizer's k-means centroids are fitted, not trained.</remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        _network?.Units.DeclareCentroids(state, "SpeechUnitCentroids");
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes, without dropout.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = _network ?? throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return BatchLoss(network, input, target, training: false)[0];
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "VoxtLM-Native" : "VoxtLM-ONNX", Description = "VoxtLM: a decoder-only language model over HuBERT units and text (Maiti et al., 2024)", FeatureCount = _options.ModelDim, Complexity = _options.NumLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VoxtLM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
