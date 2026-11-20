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

namespace AiDotNet.SpeechRecognition.Streaming;

/// <summary>
/// Emformer RNN-T (Shi et al. 2021): a streaming Transformer transducer whose attention for each segment covers a bounded
/// left context, the segment and a short look-ahead.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Emformer RNN-T listens to audio in short segments, so it can transcribe while you speak: each
/// segment sees a little of the past and a few frames of the future, and a small recurrent network turns the encoded audio
/// into text token by token.</para>
/// <para>
/// The network is torchaudio's <c>emformer_rnnt_base</c>: 80-dimensional log-mel frames through a bias-free linear layer to
/// 128 dimensions, time reduction by stacking 4 frames (width 512), 20 Emformer layers with 8 heads, 2048 GELU feed-forward
/// units, segments of 16 frames (4 after reduction), a 4-frame look-ahead (1 after reduction) and 30 frames of left context,
/// an output linear layer to 1024 with layer norm, a prediction network of a 512-dimensional symbol embedding and three
/// layer-normalized LSTMs of 512 units between layer norms, and a joiner applying ReLU to the summed encodings. Each
/// layer processes the look-ahead frames as separate right-context rows, as the paper's training does; the memory bank
/// is off, as in the base configuration. Training uses the transducer loss over SpecAugmented features.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 4097);
/// var model = new EmformerRNNT&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition", "https://arxiv.org/abs/2010.10759", Year = 2021, Authors = "Shi et al.")]
[PaperOptimizer(OptimizerKind.Adam,
                Source = "Shi et al. 2021, Sec. 4: all hybrid models are trained with the Adam "
                        + "optimizer over 180 epochs. The paper states no learning rate, batch size or "
                        + "schedule, so none is declared.")]
public partial class EmformerRNNT<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The output vocabulary, blank (0) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly EmformerRNNTOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private EmformerRnntNetwork<T>? _network;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal EmformerRnntNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public EmformerRNNT(NeuralNetworkArchitecture<T> architecture, string modelPath, EmformerRNNTOptions? options = null) : base(architecture)
    {
        _options = options ?? new EmformerRNNTOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public EmformerRNNT(NeuralNetworkArchitecture<T> architecture, EmformerRNNTOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new EmformerRNNTOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer
        ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this) ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <summary>Transcribes audio: the encoder and the transducer with greedy decoding.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for EmformerRNNT.");
        var waveform = PreprocessAudio(audio);
        List<int> tokens;
        if (_network is null) tokens = EmformerRnntNetwork<T>.GreedyCtc(Logits(waveform, training: false));
        else
        {
            using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
            var encoded = _network.Encode(_network.FeatureTensor(waveform), training: false);
            tokens = _network.GreedyTransducer(encoded);
        }
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };

    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("EmformerRNNT does not support streaming.");

    // Each token is its vocabulary entry; tokens past the vocabulary list decode as Unicode code points.
    private string Text(List<int> tokens)
    {
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (t < _options.Vocabulary.Length) { string s = _options.Vocabulary[t]; sb.Append(s == "|" ? " " : s.StartsWith("<", StringComparison.Ordinal) ? string.Empty : s); }
            else if (t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // torchaudio's emformer_rnnt_base. It used to be a generic template without Emformer attention.
        var layers = new List<LayerBase<T>>();
        _network = new EmformerRnntNetwork<T>(Engine, layers, _options.ToConfiguration());
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    private protected override int? NativeWaveformMaxAudioLengthSeconds => _network is null ? null : _options.MaxAudioLengthSeconds;

    // Per-frame log-probabilities [frames, vocabulary] of the joint network before any label is emitted, for a waveform [S], or [batch, frames, vocabulary] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input, bool training) => ForwardNativeWaveformBatch(input, _network, (net, wave) => FrameLogits(net.Encode(net.FeatureTensor(wave), training)));

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
    private int[] Labels(Tensor<T> target) => ParseCtcOrTokenTargetLabels(target, _options.VocabSize, EmformerRnntNetwork<T>.GreedyCtc);

    /// <inheritdoc />
    /// <remarks>The transducer loss (Graves 2012) of the target labels over SpecAugmented features.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target) =>
        TrainNativeWaveformBatch(input, target, _network, _optimizer, (_, row) => Labels(row), (network, wave, rowLabels) =>
        {
            var encoded = network.Encode(network.SpecAugment(network.FeatureTensor(wave), _random), training: true);
                var loss = network.TransducerLoss(encoded, rowLabels);
            return loss;
        });

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes (the transducer loss), on unaugmented features in inference
    /// mode, averaged over the batch.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) { return EvaluateNativeWaveformBatchObjective(input, target, _network, (_, row) => Labels(row), (network, wave, rowLabels) => { var features = network.FeatureTensor(wave); return network.TransducerLoss(network.Encode(features, training: false), rowLabels); }); }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "EmformerRNNT-Native" : "EmformerRNNT-ONNX", Description = "Emformer RNN-T: a streaming Transformer transducer with segment-wise attention (Shi et al., 2021)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(EmformerRNNT<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
