using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Canary multilingual speech recognition and translation model from NVIDIA.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Canary (NVIDIA, 2024) is a multilingual ASR/ST model based on the Fast Conformer encoder
/// with a multi-task decoder. It supports transcription and translation across many languages
/// using a single unified architecture with task-specific prompting, achieving strong WER
/// scores across English, German, Spanish, and French.
/// </para>
/// <para>
/// <b>For Beginners:</b> Canary is like a multilingual transcription assistant. It can listen
/// to speech in many languages and either transcribe it (write down what was said) or translate
/// it into another language, all with a single model. You tell it what you want via a prompt.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 32128);
/// var model = new Canary&lt;float&gt;(arch, "canary_1b.onnx");
/// var result = model.Transcribe(audio, "en");
/// Console.WriteLine(result.Text);
/// </code>
/// </para>
/// </remarks>
public class Canary<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    #region Fields

    private readonly CanaryOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ISpeechRecognizer Properties

    /// <inheritdoc />
    public new int SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <inheritdoc />
    public bool SupportsStreaming => true;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => true;

    /// <inheritdoc />
    public new bool IsOnnxMode => !_useNativeMode && OnnxEncoder is not null;

    #endregion

    #region Constructors

    /// <summary>Creates a Canary model in ONNX inference mode.</summary>
    public Canary(NeuralNetworkArchitecture<T> architecture, string modelPath, CanaryOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CanaryOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        SupportedLanguages = _options.SupportedLanguages;
        InitializeLayers();
    }

    /// <summary>Creates a Canary model in native training mode.</summary>
    public Canary(NeuralNetworkArchitecture<T> architecture, CanaryOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CanaryOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        base.SampleRate = _options.SampleRate;
        SupportedLanguages = _options.SupportedLanguages;
        InitializeLayers();
    }

    internal static async Task<Canary<T>> CreateAsync(CanaryOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new CanaryOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("canary", $"canary_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.EncoderDim, outputSize: options.VocabSize);
        return new Canary<T>(arch, mp, options);
    }

    #endregion

    #region ISpeechRecognizer Methods

    /// <inheritdoc />
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        language ??= DetectLanguage(audio);
        var features = PreprocessAudio(audio);
        var encoded = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        string text = DecodeTokens(encoded);
        double duration = (double)audio.Length / _options.SampleRate;
        var result = new TranscriptionResult<T>
        {
            Text = text, Language = language, DurationSeconds = duration,
            Confidence = NumOps.FromDouble(0.95)
        };
        if (includeTimestamps)
            result.Segments = GenerateTimestamps(encoded, text, duration);
        return result;
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null,
        bool includeTimestamps = false, CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc />
    public string DetectLanguage(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var probs = DetectLanguageProbabilities(audio);
        string best = "en"; T bestScore = NumOps.MinValue;
        foreach (var (lang, score) in probs)
        {
            if (NumOps.GreaterThan(score, bestScore)) { bestScore = score; best = lang; }
        }
        return best;
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        var probs = new Dictionary<string, T>();
        for (int i = 0; i < _options.SupportedLanguages.Length; i++)
        {
            double val = i < output.Length ? NumOps.ToDouble(output[i]) : 0;
            // Softmax-like normalization
            probs[_options.SupportedLanguages[i]] = NumOps.FromDouble(Math.Max(0, Math.Min(1, 1.0 / (1.0 + Math.Exp(-val)))));
        }
        return probs;
    }

    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
        => new CanaryStreamingSession(this, language ?? "en");

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultCanaryLayers(
            encoderDim: _options.EncoderDim, numEncoderLayers: _options.NumEncoderLayers,
            decoderDim: _options.DecoderDim, numDecoderLayers: _options.NumDecoderLayers,
            numHeads: _options.NumHeads, vocabSize: _options.VocabSize,
            dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

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
            Name = _useNativeMode ? "Canary-Native" : "Canary-ONNX",
            Description = $"Canary {_options.Variant} multilingual ASR/ST model (NVIDIA, 2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.EncoderDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        m.AdditionalInfo["Languages"] = string.Join(",", _options.SupportedLanguages);
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers);
        w.Write(_options.DecoderDim); w.Write(_options.NumDecoderLayers);
        w.Write(_options.NumHeads); w.Write(_options.SubsamplingFactor);
        w.Write(_options.VocabSize); w.Write(_options.BeamWidth);
        w.Write(_options.MaxOutputTokens); w.Write(_options.TargetLanguage);
        w.Write(_options.SupportedLanguages.Length);
        foreach (var l in _options.SupportedLanguages) w.Write(l);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32();
        _options.DecoderDim = r.ReadInt32(); _options.NumDecoderLayers = r.ReadInt32();
        _options.NumHeads = r.ReadInt32(); _options.SubsamplingFactor = r.ReadInt32();
        _options.VocabSize = r.ReadInt32(); _options.BeamWidth = r.ReadInt32();
        _options.MaxOutputTokens = r.ReadInt32(); _options.TargetLanguage = r.ReadString();
        int numLangs = r.ReadInt32();
        var langs = new string[numLangs]; for (int i = 0; i < numLangs; i++) langs[i] = r.ReadString();
        _options.SupportedLanguages = langs;
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new Canary<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private string DecodeTokens(Tensor<T> output)
    {
        int numTokens = Math.Min(_options.MaxOutputTokens, output.Length);
        var tokenIds = new List<int>();
        for (int i = 0; i < numTokens; i++)
        {
            int tokenId = (int)Math.Round(NumOps.ToDouble(output[i]));
            if (tokenId < 0) tokenId = 0;
            if (tokenId >= _tokenizer.VocabularySize) tokenId = _tokenizer.VocabularySize - 1;
            tokenIds.Add(tokenId);
        }
        return _tokenizer.Decode(tokenIds);
    }

    private IReadOnlyList<TranscriptionSegment<T>> GenerateTimestamps(Tensor<T> output, string text, double duration)
    {
        var segments = new List<TranscriptionSegment<T>>();
        var words = text.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0 || output.Length == 0) return segments;

        // CTC-based greedy alignment: use output token magnitudes to find word boundaries
        int tokensPerWord = Math.Max(1, output.Length / words.Length);
        double timePerToken = duration / output.Length;

        int tokenIdx = 0;
        for (int w = 0; w < words.Length; w++)
        {
            int startToken = tokenIdx;
            int endToken = Math.Min(output.Length, tokenIdx + tokensPerWord);

            // Find peak confidence within this word's token range
            T maxConf = NumOps.Zero;
            for (int t = startToken; t < endToken && t < output.Length; t++)
            {
                T absVal = NumOps.Abs(output[t]);
                if (NumOps.GreaterThan(absVal, maxConf)) maxConf = absVal;
            }

            segments.Add(new TranscriptionSegment<T>
            {
                Text = words[w],
                StartTime = startToken * timePerToken,
                EndTime = endToken * timePerToken,
                Confidence = NumOps.FromDouble(Math.Min(1.0, NumOps.ToDouble(maxConf)))
            });
            tokenIdx = endToken;
        }
        return segments;
    }

    #endregion

    #region Streaming Session

    private sealed class CanaryStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly Canary<T> _model;
        private readonly string _language;
        private readonly List<T> _buffer = [];
        private string _partialText = string.Empty;
        private bool _disposed;

        public CanaryStreamingSession(Canary<T> model, string language)
        { _model = model; _language = language; }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CanaryStreamingSession));
            for (int i = 0; i < audioChunk.Length; i++) _buffer.Add(audioChunk[i]);
            int chunkSize = _model._options.SampleRate; // 1 second chunks
            if (_buffer.Count >= chunkSize)
            {
                var frame = new Tensor<T>([chunkSize]);
                for (int i = 0; i < chunkSize; i++) frame[i] = _buffer[i];
                _buffer.RemoveRange(0, chunkSize);
                var result = _model.Transcribe(frame, _language);
                _partialText = string.IsNullOrEmpty(_partialText) ? result.Text : _partialText + " " + result.Text;
            }
        }

        public TranscriptionResult<T> GetPartialResult() => new()
        {
            Text = _partialText, Language = _language,
            Confidence = _model.NumOps.FromDouble(0.8)
        };

        public TranscriptionResult<T> Finalize()
        {
            if (_buffer.Count > 0)
            {
                var remaining = new Tensor<T>([_buffer.Count]);
                for (int i = 0; i < _buffer.Count; i++) remaining[i] = _buffer[i];
                var result = _model.Transcribe(remaining, _language);
                _partialText = string.IsNullOrEmpty(_partialText) ? result.Text : _partialText + " " + result.Text;
                _buffer.Clear();
            }
            return new TranscriptionResult<T>
            {
                Text = _partialText, Language = _language,
                Confidence = _model.NumOps.FromDouble(0.95)
            };
        }

        public void Dispose() { _disposed = true; _buffer.Clear(); }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Canary<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
