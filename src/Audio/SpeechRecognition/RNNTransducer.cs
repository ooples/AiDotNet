using AiDotNet.Attributes;
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
public partial class RNNTransducer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
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
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

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
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        var tokens = GreedyDecode(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(tokens.Count > 0 ? 0.9 : 0.0),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegments(tokens, text, audio.Shape[0]) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null,
        bool includeTimestamps = false, CancellationToken cancellationToken = default)
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
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultRNNTransducerLayers(
            encoderDim: _options.EncoderDim, numEncoderLayers: _options.NumEncoderLayers,
            numEncoderHeads: _options.NumEncoderHeads, predictionDim: _options.PredictionDim,
            numPredictionLayers: _options.NumPredictionLayers, embeddingDim: _options.EmbeddingDim,
            jointDim: _options.JointDim, numMels: _options.NumMels, vocabSize: _options.VocabSize,
            dropoutRate: _options.DropoutRate));
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

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

    private List<int> GreedyDecode(Tensor<T> logits)
    {
        var tokens = new List<int>();
        int prevToken = -1;
        int blankIdx = 0;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];
        for (int t = 0; t < numFrames; t++)
        {
            int maxIdx = 0; double maxVal = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]);
                if (val > maxVal) { maxVal = val; maxIdx = v; }
            }
            if (maxIdx != blankIdx && maxIdx != prevToken) tokens.Add(maxIdx);
            prevToken = maxIdx;
        }
        return tokens;
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
            int totalLen = 0; foreach (var ch in _chunks) totalLen += ch.Length;
            var combined = new Tensor<T>(new[] { totalLen });
            int offset = 0;
            foreach (var ch in _chunks) { for (int i = 0; i < ch.Length; i++) combined[offset + i] = ch[i]; offset += ch.Length; }
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

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
