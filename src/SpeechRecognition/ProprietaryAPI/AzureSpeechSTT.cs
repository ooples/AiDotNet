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

namespace AiDotNet.SpeechRecognition.ProprietaryAPI;

/// <summary>
/// Azure Speech's nearest disclosed recognizer: Microsoft's streaming Conformer transducer (Chen et al. 2021), a stand-in for the hosted service.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This model turns 16 kHz audio into an 80-band log-mel spectrogram, shortens it 4x with two convolutions and encodes it with Conformer blocks that attend only within short chunks plus a bounded history, so it can run as audio arrives. A two-layer LSTM remembers the tokens written so far and a joint network combines it with each audio frame to pick the next token or "nothing yet".</para>
/// <para>
/// This is a stand-in, not Microsoft's production Azure model, whose architecture and weights are not published. It is the nearest architecture Microsoft has disclosed for streaming recognition at Azure scale: the Conformer transducer (C-T) of Chen et al. 2021, section 4.1, with chunk-wise attention over a truncated history (section 3.2). From the paper: a Conformer encoder of width 640 with a 3-wide depthwise kernel, a prediction network of two LSTM layers of 720 units, a joint network, 4,000 sentence pieces, and chunks of 24 frames of 30 ms (720 ms) attending 60 frames (1.8 s) of history. Inferred: 18 layers and 1,024 feed-forward units (the paper gives them for its Transformer transducer and sizes C-T to match it), 8 heads (not stated), and a 640-wide joint network. The paper stacks eight 10 ms frames into a 30 ms step; this uses 4x convolution subsampling, so a chunk is 18 frames of 40 ms (the same 720 ms), and the history is 3 chunks (2.16 s, the nearest whole number of chunks above 1.8 s). The paper's blocks use relative positional encoding; these use rotary position encoding. The paper gives no optimizer, so the default is AdamW.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 34);
/// var model = new AzureSpeechSTT&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Developing Real-time Streaming Transformer Transducer for Speech Recognition on Large-scale Dataset", "https://arxiv.org/abs/2010.11395", Year = 2021, Authors = "Xie Chen, Yu Wu, Zhenghao Wang, Shujie Liu, Jinyu Li")]
[AiDotNet.Attributes.PreprocessesInput("ConformerAsrNetwork converts the waveform to log-mel features and feeds them to the first convolution as a [B, 1, frames, bins] image, so Layers[0] never sees the raw samples.")]
[AiDotNet.Attributes.StackInputLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Channels, AiDotNet.Enums.TensorAxis.Height, AiDotNet.Enums.TensorAxis.Width)]
public partial class AzureSpeechSTT<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The output vocabulary, blank (0) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly AzureSpeechSTTOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private ConformerAsrNetwork<T>? _network;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal ConformerAsrNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => true;
    public bool SupportsWordTimestamps => false;

    public AzureSpeechSTT(NeuralNetworkArchitecture<T> architecture, string modelPath, AzureSpeechSTTOptions? options = null) : base(architecture)
    {
        _options = options ?? new AzureSpeechSTTOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public AzureSpeechSTT(NeuralNetworkArchitecture<T> architecture, AzureSpeechSTTOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new AzureSpeechSTTOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer
        ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this) ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <summary>Transcribes audio: the encoder and the transducer with greedy decoding.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for AzureSpeechSTT.");
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

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };

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
        // The stand-in network (see the remarks).
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

    /// <inheritdoc />
    /// <remarks>The transducer loss (Graves 2012) of the target labels over SpecAugmented features (Gulati et al. 2020, section 3.1).</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows, perRow = target.Length / rows;
        var labels = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            var row = rows == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
                target.Rank >= 3 ? target.Shape.ToArray().Skip(1).ToArray() : new[] { perRow });
            labels[r] = Labels(row);
        }
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
        }, _optimizer);
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
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows, perRow = target.Length / rows;
        double total = 0;
        for (int r = 0; r < rows; r++)
        {
            var row = rows == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
                target.Rank >= 3 ? target.Shape.ToArray().Skip(1).ToArray() : new[] { perRow });
            var features = network.Features(Samples(input, r * samples, samples));
            total += NumOps.ToDouble(network.TransducerLoss(network.Encode(features, training: false), Labels(row))[0]);
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
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "AzureSpeechSTT-Native" : "AzureSpeechSTT-ONNX", Description = "Azure Speech stand-in: Microsoft's streaming Conformer transducer (Chen et al. 2021)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AzureSpeechSTT<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    // Collects the audio fed to it and transcribes the buffer on each partial result.
    private sealed class BufferedSession : IStreamingTranscriptionSession<T>
    {
        private readonly AzureSpeechSTT<T> _model;
        private readonly string _language;
        private readonly List<T> _audio = new();
        private bool _disposed;

        public BufferedSession(AzureSpeechSTT<T> model, string language) { _model = model; _language = language; }

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

        public void Dispose() => _disposed = true;
    }
}
