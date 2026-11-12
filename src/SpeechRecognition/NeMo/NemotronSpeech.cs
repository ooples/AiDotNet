using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Foundations;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>
/// Nemotron Speech Streaming: NVIDIA's cache-aware Fast Conformer (Noroozi et al. 2024) with an RNN-T decoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Nemotron Speech transcribes English as it is spoken. Its encoder never looks further
/// ahead than a fixed lookahead: the subsampling convolutions and each block's convolution see only the past, and each
/// frame's attention covers its own chunk and a fixed window of chunks before it. A transducer decoder then writes
/// tokens frame by frame, so the transcript can grow while audio is still arriving.</para>
/// <para>
/// The defaults are the released nvidia/nemotron-speech-streaming-en-0.6b (its <c>model_config.yaml</c>): 128 log-mel
/// bands without normalization, 8x causal depthwise-separable subsampling of 256 channels to 80 ms frames, 24 Conformer
/// blocks of width 1,024 (8 heads, feed-forward 4,096, a causal kernel-9 convolution), chunked-limited attention with 70
/// frames of left context and a 13-frame lookahead (a 14-frame chunk, 1.12 s), and an RNN-T decoder of a two-layer
/// 640-unit LSTM prediction network and a 640-wide joint network over a 1,024-piece SentencePiece vocabulary.
/// SpecAugment masks two frequency bands of up to 27 bins and ten time spans of 5%. Training uses AdamW (betas 0.9 and
/// 0.98, weight decay 1e-3) under the Noam schedule with factor 3.0 at width 1,024 and 8,000 warm-up steps. The
/// release's blocks use relative positional encoding; these use rotary position encoding, the relative scheme
/// <see cref="AiDotNet.NeuralNetworks.Layers.ConformerBlockLayer{T}"/> provides. The release's blank is its last id;
/// here it is 0, and piece <c>p</c> is id <c>p + 1</c>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 1025);
/// var model = new NemotronSpeech&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition", "https://arxiv.org/abs/2312.17279", Year = 2024, Authors = "Noroozi et al.")]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.98, WeightDecay = 1e-3,
                Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 8000, MinLearningRate = 1e-6,
                Source = "nvidia/nemotron-speech-streaming-en-0.6b model_config.yaml: adamw, betas [0.9, 0.98], weight_decay 0.001, lr 3.0 under NoamAnnealing with d_model 1024, warmup_steps 8000 and min_lr 1e-6. No learning rate is declared because Noam's peak is the factor times a function of the width; the model builds the schedule with that factor itself.")]
[AiDotNet.Attributes.PreprocessesInput("ConformerAsrNetwork converts the waveform to log-mel features and feeds them to the first convolution as a [B, 1, frames, bins] image, so Layers[0] never sees the raw samples.")]
[AiDotNet.Attributes.StackInputLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Channels, AiDotNet.Enums.TensorAxis.Height, AiDotNet.Enums.TensorAxis.Width)]
public partial class NemotronSpeech<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    private const string TokenizerResource = "AiDotNet.Tokenization.nemotron-speech-1024.model.gz";
    private static readonly Lazy<SentencePieceModel> ReleaseText = new(LoadTokenizer);

    /// <inheritdoc />
    /// <remarks>The output vocabulary, blank (0) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly NemotronSpeechOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The release network on the native path; null in ONNX mode or with custom layers.
    private ConformerAsrNetwork<T>? _network;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal ConformerAsrNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public NemotronSpeech(NeuralNetworkArchitecture<T> architecture, string modelPath, NemotronSpeechOptions? options = null) : base(architecture)
    {
        _options = options ?? new NemotronSpeechOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public NemotronSpeech(NeuralNetworkArchitecture<T> architecture, NemotronSpeechOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new NemotronSpeechOptions(); Options = _options; _useNativeMode = true;
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer ?? ReleaseOptimizer());
    }

    /// <inheritdoc />
    // Rebuilt from the options if it was cleared (a probe that changes an option clears it).
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer =>
        _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseOptimizer());

    // The release's AdamW under Noam with its factor, which the paper-optimizer factory's Noam does not take.
    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer() => new(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
    {
        Beta1 = 0.9, Beta2 = 0.98, WeightDecay = _options.WeightDecay,
        LearningRateScheduler = new NoamSchedule(_options.EncoderDim, Math.Max(1, _options.WarmupSteps), _options.NoamFactor),
        SchedulerStepMode = SchedulerStepMode.StepPerBatch,
    });

    /// <summary>The release's SentencePiece tokenizer (1,024 pieces; output id <c>p + 1</c> is piece <c>p</c>).</summary>
    public static SentencePieceModel Tokenizer => ReleaseText.Value;

    private static SentencePieceModel LoadTokenizer()
    {
        using var stream = typeof(NemotronSpeech<T>).Assembly.GetManifestResourceStream(TokenizerResource)
            ?? throw new InvalidDataException($"The embedded resource {TokenizerResource} is missing.");
        return SentencePieceModel.FromGzip(stream);
    }

    /// <summary>Transcribes audio: the encoder and the transducer with greedy decoding.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for NemotronSpeech.");
        var waveform = PreprocessAudio(audio);
        List<int> tokens;
        if (_network is null) tokens = ConformerAsrNetwork<T>.GreedyCtc(Logits(waveform, training: false));
        else
        {
            using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
            var encoded = _network.Encode(_network.Features(Samples(waveform, 0, waveform.Length)), training: false);
            tokens = _network.GreedyTransducer(encoded);
        }
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };

    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("NemotronSpeech does not support streaming sessions.");

    // A supplied vocabulary names each id; otherwise the release's pieces (id - 1) when the vocabulary is the release's,
    // and Unicode code points for any other size.
    private string Text(List<int> tokens)
    {
        if (_options.Vocabulary is { } vocabulary)
            return string.Concat(tokens.Where(t => t < vocabulary.Length).Select(t => vocabulary[t])).Trim();
        if (_options.VocabSize == Tokenizer.Count + 1) return Tokenizer.Decode(tokens.Select(t => t - 1)).Trim();
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens) if (t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t));
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The release network (see the class remarks). It used to be a generic encoder-LLM template.
        var layers = new List<LayerBase<T>>();
        _network = new ConformerAsrNetwork<T>(Engine, layers, _options.ToConfiguration());
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_network is null) return base.PreprocessAudio(rawAudio);
        int length = Math.Min(rawAudio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private double[] Samples(Tensor<T> waveform, int offset, int count)
    {
        var samples = new double[count];
        var span = waveform.Data.Span;
        for (int i = 0; i < count; i++) samples[i] = NumOps.ToDouble(span[offset + i]);
        return samples;
    }

    // Per-frame log-probabilities [frames, vocabulary] of the joint network before any label is emitted, for a waveform [S], or [batch, frames, vocabulary] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++) outputs[r] = FrameLogits(_network.Encode(_network.Features(Samples(input, r * samples, samples)), training));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    private Tensor<T> FrameLogits(Tensor<T> encoded)
    {
        var network = _network ?? throw new InvalidOperationException("No native network.");
        var joint = network.Joint(encoded, network.Predict(Array.Empty<int>()));
        return Engine.Reshape(joint, new[] { encoded.Shape[0], _options.VocabSize });
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    // The labels a training target names: label ids [length], or one distribution per encoder frame [frames, vocabulary]
    // read as a CTC alignment (the most likely symbol per frame, repeats merged, blanks removed).
    private int[] Labels(Tensor<T> target)
    {
        if (target.Rank >= 2 && target.Shape[^1] == _options.VocabSize)
        {
            var frames = Engine.Reshape(target, new[] { target.Length / _options.VocabSize, _options.VocabSize });
            return ConformerAsrNetwork<T>.GreedyCtc(frames).ToArray();
        }
        var labels = new int[target.Length];
        var span = target.Data.Span;
        for (int i = 0; i < labels.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value <= 0 || value >= _options.VocabSize) throw new ArgumentException($"Label {value} is the blank or outside the vocabulary.", nameof(target));
            labels[i] = (int)value;
        }
        return labels;
    }

    private Tensor<T> Row(Tensor<T> target, int rows, int r)
    {
        int perRow = target.Length / rows;
        return rows == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
            target.Rank >= 3 ? target.Shape.ToArray().Skip(1).ToArray() : new[] { perRow });
    }

    /// <inheritdoc />
    /// <remarks>The transducer loss (Graves 2012) with FastEmit of the target labels over SpecAugmented features.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var labels = new int[rows][];
        for (int r = 0; r < rows; r++) labels[r] = Labels(Row(target, rows, r));
        TrainWithCustomLoss(input, _ =>
        {
            Tensor<T>? total = null;
            for (int r = 0; r < rows; r++)
            {
                var encoded = network.Encode(network.SpecAugment(network.Features(Samples(input, r * samples, samples)), _random), training: true);
                var loss = network.TransducerLoss(encoded, labels[r]);
                total = total is null ? loss : Engine.TensorAdd(total, loss);
            }
            return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
        });
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes (the transducer loss), on unaugmented features in inference
    /// mode, averaged over the batch.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = _network ?? throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        double total = 0;
        for (int r = 0; r < rows; r++)
        {
            var features = network.Features(Samples(input, r * samples, samples));
            total += NumOps.ToDouble(network.TransducerLoss(network.Encode(features, training: false), Labels(Row(target, rows, r)))[0]);
        }
        return NumOps.FromDouble(total / rows);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "NemotronSpeech-Native" : "NemotronSpeech-ONNX", Description = "Nemotron Speech Streaming: a cache-aware Fast Conformer with an RNN-T decoder", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(NemotronSpeech<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
