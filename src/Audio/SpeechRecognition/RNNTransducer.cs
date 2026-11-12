using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// RNN-Transducer (RNN-T) streaming speech recognition model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RNN-Transducer (Graves, 2012; He et al., 2019) combines an audio encoder with a label
/// prediction network and a joint network to produce a streaming ASR system. Unlike CTC,
/// RNN-T can model output dependencies through its prediction network, achieving strong
/// results on LibriSpeech (WER 2.0% with LM) without an external language model. It is
/// the backbone of on-device ASR in Google's Pixel phones and NVIDIA Riva.
/// </para>
/// <para>
/// <b>For Beginners:</b> RNN-T is a real-time speech recognizer ideal for live transcription.
/// Unlike batch models (like Whisper) that need the whole audio, RNN-T processes speech as
/// it arrives - perfect for live captioning and voice assistants.
///
/// It has three parts:
/// 1. <b>Encoder</b> - Listens to the audio (converts sound to features)
/// 2. <b>Predictor</b> - Remembers what was already said (like a small language model)
/// 3. <b>Joiner</b> - Combines both to decide the next output token
///
/// Think of it like a court stenographer who listens (encoder), remembers the context
/// (predictor), and types the next word (joiner) - all in real time.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 5000);
/// var model = new RNNTransducer&lt;float&gt;(arch, "rnnt_medium.onnx");
/// var result = model.Transcribe(audioWaveform);
/// // Result is available in the returned value // "hello world"
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Sequence Transduction with Recurrent Neural Networks", "https://arxiv.org/abs/1211.3711", Year = 2012, Authors = "Alex Graves")]
[PaperOptimizer(OptimizerKind.SgdMomentum, LearningRate = 1e-4, Momentum = 0.9,
                Source = "Graves 2012: a learning rate of 1e-4 and a momentum of 0.9 with steepest descent.")]
public partial class RNNTransducer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Measured from the output construction: <c>PredictCore</c> folds the whole <c>Layers</c> chain and
    /// <c>PostprocessOutput</c> is the identity, so the width is the final layer's output dimension.
    /// <c>LayerHelper.CreateDefaultRNNTransducerLayers</c> ends its joint network with
    /// <c>new FullyConnectedLayer&lt;T&gt;(vocabSize, null)</c>, and <c>InitializeLayers</c> passes
    /// <c>vocabSize: _options.VocabSize</c>. Note it is the joint output, not <c>JointDim</c> - that
    /// is the hidden layer one step earlier.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    #region Fields

    private readonly RNNTransducerOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // The transducer on the native path: transcription (encoder), prediction and joint networks. Null in ONNX
    // mode or with custom layers.
    private ConformerAsrNetwork<T>? _network;

    // Dropout on the transcription network's output while training (null when DropoutRate is zero).
    private DropoutLayer<T>? _encoderDropout;

    // At most this many labels are emitted at one frame before the search moves on.
    private const int MaxSymbolsPerFrame = 10;

    internal ConformerAsrNetwork<T>? Network => _network;

    #endregion

    #region ISpeechRecognizer Properties

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <inheritdoc />
    public bool SupportsStreaming => true;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => true;

    #endregion

    #region Constructors

    /// <summary>Creates an RNN-T model in ONNX inference mode.</summary>
    public RNNTransducer(NeuralNetworkArchitecture<T> architecture, string modelPath, RNNTransducerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new RNNTransducerOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    /// <summary>Creates an RNN-T model in native training mode.</summary>
    public RNNTransducer(NeuralNetworkArchitecture<T> architecture, RNNTransducerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new RNNTransducerOptions();
        Options = _options;
        _useNativeMode = true;
        // The rate this model publishes on its own options. Built bare, the optimizer
        // would use its own default instead and LearningRate would be configuration that
        // nothing reads — the defect that diverged MusicFlamingo's training.
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
        new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        { InitialLearningRate = _options.LearningRate });
        base.SampleRate = _options.SampleRate;
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    internal static async Task<RNNTransducer<T>> CreateAsync(RNNTransducerOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new RNNTransducerOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("rnnt", $"rnnt_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.VocabSize);
        return new RNNTransducer<T>(arch, mp, options);
    }

    #endregion

    #region ISpeechRecognizer

    /// <inheritdoc />
    /// <remarks>Greedy transducer decoding (Graves 2012, section 2.2): at each frame the most likely symbol is emitted
    /// and fed back to the prediction network until the blank moves to the next frame.</remarks>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
        => Decode(audio, beamWidth: 1, language, includeTimestamps);

    /// <summary>
    /// Transcribes audio with the transducer beam search of Graves 2012 (Algorithm 1): hypotheses over the
    /// (frame, label) lattice merged across alignments, keeping <paramref name="beamWidth"/> per frame and
    /// returning the one with the highest log-probability per label.
    /// </summary>
    /// <param name="audio">The waveform.</param>
    /// <param name="beamWidth">The hypotheses kept per frame; 1 is greedy decoding.</param>
    /// <param name="language">The language reported in the result.</param>
    /// <param name="includeTimestamps">Whether to return segments.</param>
    /// <returns>The transcription.</returns>
    public TranscriptionResult<T> TranscribeWithBeamSearch(Tensor<T> audio, int beamWidth, string? language = null, bool? includeTimestamps = null)
    {
        if (beamWidth < 1) throw new ArgumentOutOfRangeException(nameof(beamWidth), "The beam must keep at least one hypothesis.");
        return Decode(audio, beamWidth, language, includeTimestamps);
    }

    private TranscriptionResult<T> Decode(Tensor<T> audio, int beamWidth, string? language, bool? includeTimestamps)
    {
        ThrowIfDisposed();
        var waveform = PreprocessAudio(audio);
        List<int> tokens;
        if (_network is null)
        {
            // An ONNX graph or custom layers carry no separate prediction network to step, so their frame
            // outputs are read per frame.
            var logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(waveform) : Predict(waveform);
            tokens = SpeechRecognitionEngineOps.CollapseCtcFrames(SpeechRecognitionEngineOps.FrameArgMax(logits), 0);
        }
        else
        {
            tokens = DecodeTransducer(_network, waveform, beamWidth);
        }

        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(tokens.Count > 0 ? 0.9 : 0.0),
            DurationSeconds = duration,
            Segments = ResolveReturnTimestamps(includeTimestamps) ? ExtractSegments(tokens, text, audio.Shape[0]) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null,
        bool? includeTimestamps = null, CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc />
    public string DetectLanguage(Tensor<T> audio) => _options.Language;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
        => new Dictionary<string, T> { [_options.Language] = NumOps.FromDouble(1.0) };

    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
        => new RNNTStreamingSession(this, language ?? _options.Language);

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        _network = null;
        _encoderDropout = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }

        // Graves 2012, section 2: a transcription network over log-mel frames, a prediction network (label
        // embedding then LSTMs) over the labels emitted so far, and a joint network combining the two into
        // Pr(k | t, u). The transcription network is a Conformer stack after 4x convolutional subsampling.
        var layers = new List<LayerBase<T>>();
        _network = new ConformerAsrNetwork<T>(Engine, layers, new ConformerAsrConfiguration
        {
            SampleRate = _options.SampleRate, NumMels = _options.NumMels, EncoderDim = _options.EncoderDim,
            NumEncoderLayers = _options.NumEncoderLayers, NumAttentionHeads = _options.NumEncoderHeads,
            Decoder = ConformerAsrDecoder.Transducer, PredictionDim = _options.PredictionDim,
            PredictionLayers = _options.NumPredictionLayers, PredictionEmbeddingDim = _options.EmbeddingDim,
            JointDim = _options.JointDim, VocabSize = _options.VocabSize, FrequencyMasks = 0, TimeMasks = 0,
        });
        Layers.AddRange(layers);
        if (_options.DropoutRate > 0)
        {
            _encoderDropout = new DropoutLayer<T>(_options.DropoutRate);
            Layers.Add(_encoderDropout);
        }
    }

    /// <inheritdoc />
    /// <remarks>Native: the joint network's log-probabilities <c>[frames, VocabSize]</c> before any label is emitted
    /// (u = 0), for a waveform <c>[samples]</c>, or <c>[batch, frames, VocabSize]</c> for <c>[batch, samples]</c>.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        return Logits(input, training: false);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    /// <inheritdoc />
    /// <remarks>The transducer loss (Graves 2012, section 2.1): the negative log-probability of the target labels summed
    /// over every alignment of the lattice, averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var labels = RowLabels(target, rows);
        TrainWithCustomLoss(input, _ => MeanTransducerLoss(network, input, labels, training: true), _optimizer);
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes, in inference mode.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = _network ?? throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        return MeanTransducerLoss(network, input, RowLabels(target, rows), training: false).GetFlat(0);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "RNN-T-Native" : "RNN-T-ONNX",
            Description = $"RNN-Transducer {_options.Variant} streaming ASR (Graves, 2012; He et al., 2019)",
            Complexity = _options.NumEncoderLayers + _options.NumPredictionLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["EncoderDim"] = _options.EncoderDim.ToString();
        m.AdditionalInfo["PredictionDim"] = _options.PredictionDim.ToString();
        m.AdditionalInfo["JointDim"] = _options.JointDim.ToString();
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        return m;
    }





    #endregion

    #region Private Helpers

    // One waveform row of the input: [samples] for a single utterance, row r of [batch, samples] otherwise.
    private Tensor<T> Row(Tensor<T> input, int row, int rows)
    {
        int samples = input.Length / rows;
        return rows == 1 && input.Rank == 1
            ? input
            : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(input, new[] { rows, samples }), new[] { row, 0 }, new[] { 1, samples }), new[] { samples });
    }

    // The joint network's log-probabilities before any label, per frame.
    private Tensor<T> Logits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        var network = _network;
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
        {
            var encoded = Encoded(network, Row(input, r, rows), training);
            var joint = network.Joint(encoded, network.Predict(Array.Empty<int>()));
            outputs[r] = Engine.Reshape(joint, new[] { encoded.Shape[0], _options.VocabSize });
        }
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    // The transcription network's output [frames, EncoderDim], with dropout while training.
    private Tensor<T> Encoded(ConformerAsrNetwork<T> network, Tensor<T> waveform, bool training)
    {
        var encoded = network.Encode(network.FeatureTensor(waveform), training);
        return training && _encoderDropout is not null ? _encoderDropout.Forward(encoded) : encoded;
    }

    private Tensor<T> MeanTransducerLoss(ConformerAsrNetwork<T> network, Tensor<T> input, int[][] labels, bool training)
    {
        int rows = labels.Length;
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var loss = network.TransducerLoss(Encoded(network, Row(input, r, rows), training), labels[r]);
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    private int[][] RowLabels(Tensor<T> target, int rows)
    {
        int perRow = target.Length / rows;
        var labels = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            var row = rows == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
                target.Rank >= 3 ? target.Shape.ToArray().Skip(1).ToArray() : new[] { perRow });
            labels[r] = Labels(row);
        }
        return labels;
    }

    // Target labels: label ids [length], or one distribution per frame [frames, VocabSize] read as a CTC alignment.
    private int[] Labels(Tensor<T> target)
    {
        if (target.Rank >= 2 && target.Shape[^1] == _options.VocabSize)
        {
            var frames = Engine.Reshape(target, new[] { target.Length / _options.VocabSize, _options.VocabSize });
            return SpeechRecognitionEngineOps.CollapseCtcFrames(SpeechRecognitionEngineOps.FrameArgMax(frames), 0).ToArray();
        }

        // Round in T, convert the whole tensor to ids once, then validate the ids as ints.
        var labels = Engine.TensorRound(target).Cast<int>().ToArray();
        foreach (int value in labels)
        {
            if (value <= 0 || value >= _options.VocabSize) throw new ArgumentException($"Label {value} is the blank or outside the vocabulary.", nameof(target));
        }
        return labels;
    }

    // Greedy or beam transducer decoding over the encoded utterance (Graves 2012, section 2.2).
    private List<int> DecodeTransducer(ConformerAsrNetwork<T> network, Tensor<T> waveform, int beamWidth)
    {
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        int rows = waveform.Rank == 1 ? 1 : waveform.Shape[0];
        var encoded = Encoded(network, Row(waveform, 0, rows), training: false);
        int frames = encoded.Shape[0], width = encoded.Shape[1], vocabulary = _options.VocabSize;

        // log Pr(k | t, u) after the prediction network has read the labels.
        Tensor<T> LogProbs(IReadOnlyList<int> history, int t)
        {
            var predicted = network.Predict(history);
            var last = Engine.TensorSlice(predicted, new[] { history.Count, 0 }, new[] { 1, predicted.Shape[1] });
            var frame = Engine.TensorSlice(encoded, new[] { t, 0 }, new[] { 1, width });
            return Engine.Reshape(network.Joint(frame, last), new[] { vocabulary });
        }

        if (beamWidth <= 1) return TransducerSearch.Greedy<T>(frames, LogProbs, 0, MaxSymbolsPerFrame);
        var hypotheses = TransducerSearch.BeamSearch<T>(frames, LogProbs, vocabulary, 0, beamWidth, MaxSymbolsPerFrame);
        return hypotheses.Count > 0 ? hypotheses[0].Labels.ToList() : new List<int>();
    }

    private string TokensToText(List<int> tokens)
    {
        var vocab = _options.Vocabulary;
        var chars = new List<char>();
        foreach (var token in tokens)
        {
            if (token >= 0 && token < vocab.Length)
            {
                var symbol = vocab[token];
                if (symbol == "|" || symbol == " ") chars.Add(' ');
                else if (symbol.Length == 1 && char.IsLetter(symbol[0])) chars.Add(symbol[0]);
                else if (symbol == "'") chars.Add('\'');
            }
        }
        return new string(chars.ToArray()).Trim();
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(List<int> tokens, string text, int audioLength)
    {
        if (tokens.Count == 0 || string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>();
        double duration = (double)audioLength / SampleRate;
        return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(0.9) } };
    }

    #endregion

    #region Streaming Session

    private sealed class RNNTStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly RNNTransducer<T> _model;
        private readonly string _language;
        private readonly List<Tensor<T>> _chunks = new();
        private bool _disposed;

        public RNNTStreamingSession(RNNTransducer<T> model, string language) { _model = model; _language = language; }

        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(RNNTStreamingSession)); _chunks.Add(audioChunk); }

        public TranscriptionResult<T> GetPartialResult()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(RNNTStreamingSession));
            if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language };
            var combined = SpeechRecognitionEngineOps.ConcatFlat(_chunks);
            return _model.Transcribe(combined, _language);
        }

        public TranscriptionResult<T> Finalize()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(RNNTStreamingSession));
            var result = GetPartialResult();
            _disposed = true;
            return result;
        }

        public void Dispose() { _disposed = true; }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RNNTransducer<T>)); }

    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
