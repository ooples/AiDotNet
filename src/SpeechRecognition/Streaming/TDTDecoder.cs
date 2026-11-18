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
/// Token-and-Duration Transducer (Xu et al. 2023): a Conformer transducer whose joint network also predicts frame durations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> TDT turns 16 kHz audio into an 80-band log-mel spectrogram, shortens it 4x with two
/// convolutions, and encodes it with Conformer blocks, each mixing self-attention (long-range context) with a convolution
/// (local patterns). A small LSTM remembers the words written so far and a joint network combines it with each audio frame to pick the next token or "nothing yet".</para>
/// <para>
/// The paper trains Conformer TDT models in NeMo; the encoder here is a Conformer-Large (17 blocks, width 512, 8 heads, 31-wide depthwise kernel) and the decoder a 640-unit LSTM prediction network with a 640-wide joint network that predicts the token and, independently, how many frames the emission advances among 0 to 4 (the paper's best configuration, section 5). Training uses the TDT loss (Equations 3 to 5) and greedy decoding skips the predicted frames (Algorithm 1). SpecAugment masks the features in training (section 3.1). The decoder trains with the token-and-duration transducer loss and decodes greedily. The paper's blocks use relative
/// positional encoding; these use rotary position encoding, the relative scheme
/// <see cref="AiDotNet.NeuralNetworks.Layers.ConformerBlockLayer{T}"/> provides.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 34);
/// var model = new TDTDecoder&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[PreprocessesInput("ConformerAsrNetwork.Features converts each waveform row to an 80-band log-mel spectrogram, and Encode "
    + "feeds it to the 4x subsampling convolution as a [1, 1, frames, bins] image, so Layers[0] never sees the raw samples.")]
[StackInputLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width)]
[ResearchPaper("Efficient Sequence Transduction by Jointly Predicting Tokens and Durations", "https://arxiv.org/abs/2304.06795", Year = 2023, Authors = "Xu et al.")]
public partial class TDTDecoder<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The output vocabulary, blank (0) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly TDTDecoderOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private ConformerAsrNetwork<T>? _network;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal ConformerAsrNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public TDTDecoder(NeuralNetworkArchitecture<T> architecture, string modelPath, TDTDecoderOptions? options = null) : base(architecture)
    {
        _options = options ?? new TDTDecoderOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public TDTDecoder(NeuralNetworkArchitecture<T> architecture, TDTDecoderOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new TDTDecoderOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <summary>Transcribes audio: the encoder and the transducer with greedy decoding.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for TDTDecoder.");
        var waveform = PreprocessAudio(audio);
        List<int> tokens;
        if (_network is null) tokens = ConformerAsrNetwork<T>.GreedyCtc(Logits(waveform, training: false));
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

    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("TDTDecoder does not support streaming.");

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
        // The paper network (see the class remarks). It used to be a generic template.
        var layers = new List<LayerBase<T>>();
        _network = new ConformerAsrNetwork<T>(Engine, layers, _options.ToConfiguration());
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    private protected override int? NativeWaveformMaxAudioLengthSeconds => _network is null ? null : _options.MaxAudioLengthSeconds;

    // Per-frame token log-probabilities [frames, vocabulary] of the joint network before any label is emitted, for a waveform [S], or [batch, frames, vocabulary] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input, bool training) => ForwardNativeWaveformBatch(input, _network, (net, wave) => FrameLogits(net.Encode(net.FeatureTensor(wave), training)));

    private Tensor<T> FrameLogits(Tensor<T> encoded)
    {
        var network = _network ?? throw new InvalidOperationException("No native network.");
        var joint = network.Joint(encoded, network.Predict(Array.Empty<int>()));
        return Engine.Reshape(Engine.TensorSlice(joint, new[] { 0, 0, 0 }, new[] { encoded.Shape[0], 1, _options.VocabSize }), new[] { encoded.Shape[0], _options.VocabSize });
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    // The labels a training target names: label ids [length], or one distribution per encoder frame [frames, vocabulary]
    // read as a CTC alignment (the most likely symbol per frame, repeats merged, blanks removed).
    private int[] Labels(Tensor<T> target) => ParseCtcOrTokenTargetLabels(target, _options.VocabSize, ConformerAsrNetwork<T>.GreedyCtc);

    /// <inheritdoc />
    /// <remarks>The transducer loss (Graves 2012) of the target labels over SpecAugmented features (Gulati et al. 2020, section 3.1).</remarks>
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
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "TDTDecoder-Native" : "TDTDecoder-ONNX", Description = "TDT: a Conformer transducer whose joint network also predicts frame durations", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(TDTDecoder<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
