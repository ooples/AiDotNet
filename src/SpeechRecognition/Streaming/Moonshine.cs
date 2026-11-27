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
/// Moonshine Tiny (Jeffries et al. 2024): an encoder-decoder Transformer over the raw waveform with rotary positions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Moonshine Tiny shortens 16 kHz audio 384x with three convolutions, encodes it with Transformer layers, and a second Transformer writes the transcript token by token while reading the encoded audio. It is small and fast enough for live captions on a device.</para>
/// <para>
/// A three-convolution stem compresses the 16 kHz waveform 384x (widths 127, 7 and 3, strides 64, 3 and 2, with tanh and group normalization, then GELU; Figure 3), so the encoder handles audio of any length without padding. Encoder and decoder use rotary position embeddings in every layer; the encoder feed-forward networks use GELU and the decoder ones SwiGLU (Table 1). Moonshine Tiny is width 288 with 6 encoder and 6 decoder layers and 8 heads; the vocabulary is the Llama byte-level BPE (32,000 tokens) plus 768 special tokens, with start token 1 and end token 2. Training minimizes the decoder's cross-entropy and transcription decodes greedily.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Your recording: 16 kHz mono samples in [-1, 1], shaped [1, samples]. Noise stands in for it here.
/// var waveform = Tensor&lt;float&gt;.CreateRandom(1, 16000);
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 34);
/// var model = new Moonshine&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Moonshine: Speech Recognition for Live Transcription and Voice Commands", "https://arxiv.org/abs/2410.15608", Year = 2024, Authors = "Useful Sensors")]
[PaperOptimizer(OptimizerKind.ScheduleFreeAdamW, LearningRate = 1.4e-3, WarmupSteps = 8192,
                Source = "Jeffries et al. 2024: the AdamW variation of the schedule-free optimizer of Defazio et al. 2024, with gradient norm clipping and the learning rate warmed up to 1.4e-3 over 8192 steps. The clipping threshold is not stated, so none is declared.")]
[AiDotNet.Attributes.PreprocessesInput("ConformerAsrNetwork feeds the raw waveform to Moonshine's convolution stem as a single-channel [B, 1, samples] signal, so Layers[0] never sees the caller's [batch, samples] layout.")]
[AiDotNet.Attributes.StackInputLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Channels, AiDotNet.Enums.TensorAxis.Time)]
public partial class Moonshine<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The output vocabulary, blank (0) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly MoonshineOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private ConformerAsrNetwork<T>? _network;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal ConformerAsrNetwork<T>? Network => _network;
    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => true;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public Moonshine(NeuralNetworkArchitecture<T> architecture, string modelPath, MoonshineOptions? options = null) : base(architecture)
    {
        _options = options ?? new MoonshineOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public Moonshine(NeuralNetworkArchitecture<T> architecture, MoonshineOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new MoonshineOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer
        ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this) ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <summary>Transcribes audio: the encoder and the decoder with greedy decoding.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for Moonshine.");
        var waveform = PreprocessAudio(audio);
        List<int> tokens;
        if (_network is null) tokens = ConformerAsrNetwork<T>.GreedyCtc(Logits(waveform, training: false));
        else
        {
            EnsureRotaryAndCausalAttention();
            using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
            var encoded = _network.Encode(_network.FeatureTensor(waveform), training: false);
            tokens = _network.GreedyAttention(encoded, _options.MaxTextLength);
        }
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    /// <inheritdoc/>
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    /// <inheritdoc/>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };

    /// <inheritdoc/>
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new BufferedSession(this, language ?? _options.Language);

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
        // Suspend any active caller TensorArena while building persistent RoPE cos/sin caches and layer tables so
        // later per-step training arenas or clones never alias and overwrite long-lived model buffers.
        using var noArena = Tensors.Helpers.TensorArena.Suspend();
        // The paper network (see the class remarks). It used to be a generic template.
        var layers = new List<LayerBase<T>>();
        _network = new ConformerAsrNetwork<T>(Engine, layers, _options.ToConfiguration());
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    private protected override int? NativeWaveformMaxAudioLengthSeconds => _network is null ? null : _options.MaxAudioLengthSeconds;

    // Re-applies Moonshine's rotary positional encoding and causal decoder self-attention flags if a
    // clone/deserialization layer reconstruction recreated any MultiHeadAttentionLayer from constructor args alone.
    private void EnsureRotaryAndCausalAttention()
    {
        if (_network is null) return;
        var c = _network.Configuration;
        double theta = c.RopeTheta > 0 ? c.RopeTheta : 10000.0;
        if (c.RopeTheta > 0)
        {
            foreach (var block in _network.TransformerBlocks)
            {
                if (block.Attention.PositionalEncoding != PositionalEncodingType.Rotary)
                {
                    using var noArena = Tensors.Helpers.TensorArena.Suspend();
                    block.Attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, theta, 2048);
                }
            }
        }
        foreach (var block in _network.DecoderBlocks)
        {
            block.SelfAttention.UseCausalMask = true;
            if (c.DecoderRotary && block.SelfAttention.PositionalEncoding != PositionalEncodingType.Rotary)
            {
                using var noArena = Tensors.Helpers.TensorArena.Suspend();
                block.SelfAttention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, theta, 2048);
            }
        }
    }

    // The decoder's logits [1, vocabulary] for the first token of a waveform [S], or [batch, 1, vocabulary] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input, bool training) { if (_network is not null) EnsureRotaryAndCausalAttention(); return ForwardNativeWaveformBatch(input, _network, (net, wave) => FrameLogits(net.Encode(net.FeatureTensor(wave), training))); }

    private Tensor<T> FrameLogits(Tensor<T> encoded)
    {
        var network = _network ?? throw new InvalidOperationException("No native network.");
        return network.DecoderLogits(new[] { network.StartToken }, encoded);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    // The labels a training target names: label ids [length], or one distribution per encoder frame [frames, vocabulary]
    // read as a CTC alignment (the most likely symbol per frame, repeats merged, blanks removed).
    private int[] Labels(Tensor<T> target) => ParseCtcOrTokenTargetLabels(target, _options.VocabSize, ConformerAsrNetwork<T>.GreedyCtc);

    /// <inheritdoc />
    /// <remarks>The CTC loss (Graves et al. 2006) of the target labels over SpecAugmented features.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target) =>
        TrainNativeWaveformBatch(input, target, _network, _optimizer, (_, row) => Labels(row), (network, wave, rowLabels) =>
        {
            var loss = network.CtcTrainingLoss(network.SpecAugment(network.FeatureTensor(wave), _random), rowLabels, training: true);
            return loss;
        }, beforeTrain: _ => EnsureRotaryAndCausalAttention());

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes (the CTC loss), on unaugmented features in inference
    /// mode, averaged over the batch.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) { if (_network is not null) EnsureRotaryAndCausalAttention(); return EvaluateNativeWaveformBatchObjective(input, target, _network, (_, row) => Labels(row), (network, wave, rowLabels) => { var features = network.FeatureTensor(wave); return network.CtcTrainingLoss(features, rowLabels, training: false); }); }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "Moonshine-Native" : "Moonshine-ONNX", Description = "Moonshine Tiny: an encoder-decoder Transformer over the raw waveform with rotary positions", FeatureCount = 1, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Moonshine<T>)); }
    /// <inheritdoc/>
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    // Collects the audio fed to it and transcribes the buffer on each partial result.
    private sealed class BufferedSession : IStreamingTranscriptionSession<T>
    {
        private readonly Moonshine<T> _model;
        private readonly string _language;
        private readonly List<T> _audio = new();
        private bool _disposed;

        public BufferedSession(Moonshine<T> model, string language) { _model = model; _language = language; }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BufferedSession));
            _audio.AddRange(audioChunk.ToArray());
        }

        public TranscriptionResult<T> GetPartialResult()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BufferedSession));
            if (_audio.Count == 0) return new TranscriptionResult<T> { Language = _language, Confidence = MathHelper.GetNumericOperations<T>().Zero };
            return _model.Transcribe(new Tensor<T>(new[] { _audio.Count }, new Vector<T>(_audio.ToArray())), _language);
        }

        public TranscriptionResult<T> Finalize()
        {
            var result = GetPartialResult();
            _disposed = true;
            return result;
        }

        /// <inheritdoc/>
        public void Dispose() => _disposed = true;
    }
}
