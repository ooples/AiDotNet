using AiDotNet.Attributes;
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
/// // Result is available in the returned value
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.Translation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("NVIDIA Canary: An Open Multilingual Large ASR Model", "https://doi.org/10.48550/arXiv.2404.02592", Year = 2024, Authors = "Ankur Rekesh, Taejin Park, Subhankar Ghosh, Kolya Malkin, Samuel Kriman, Somshubra Majumdar, Boris Ginsburg")]
public partial class Canary<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Measured from the output construction: <c>PredictCore</c> folds the whole <c>Layers</c> chain and
    /// <c>PostprocessOutput</c> is the identity, so the width is the final layer's output dimension.
    /// <c>LayerHelper.CreateDefaultCanaryLayers</c> ends with the vocabulary projection
    /// <c>new DenseLayer&lt;T&gt;(vocabSize, null)</c>, and <c>InitializeLayers</c> passes
    /// <c>vocabSize: _options.VocabSize</c>. A token-vocabulary width, not a mel or bin count.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    #region Fields

    private readonly CanaryOptions _options;
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
        Options = _options;
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
        Options = _options;
        _useNativeMode = true;
        // The rate this model publishes on its own options. Built bare, the optimizer
        // would use its own default instead and LearningRate would be configuration that
        // nothing reads — the defect that diverged MusicFlamingo's training.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { InitialLearningRate = _options.LearningRate });
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
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
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
        if (ResolveReturnTimestamps(includeTimestamps))
            result.Segments = GenerateTimestamps(encoded, text, duration);
        return result;
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null,
        bool? includeTimestamps = null, CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc />
    public string DetectLanguage(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var probs = DetectLanguageProbabilities(audio);
        if (probs.Count == 0) return "en";
        // First language with the highest probability; NaN never wins.
        var languages = probs.Keys.ToArray();
        var scores = new Tensor<T>(probs.Values.ToArray(), [1, languages.Length]);
        var cleaned = Engine.TensorNanToNum(scores, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity);
        int best = Engine.TensorArgMax(cleaned, 1).ToArray()[0];
        return languages[best];
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        var probs = new Dictionary<string, T>();
        int languageCount = _options.SupportedLanguages.Length;
        if (languageCount == 0) return probs;

        // One sigmoid score per language from the leading outputs (missing outputs score sigmoid(0)).
        int available = Math.Min(languageCount, output.Length);
        var logits = new Tensor<T>([languageCount]);
        if (available > 0)
        {
            output.AsSpan().Slice(0, available).CopyTo(logits.AsWritableSpan());
        }

        var scores = Engine.TensorClamp(Engine.Sigmoid(logits), NumOps.Zero, NumOps.One).ToArray();
        for (int i = 0; i < languageCount; i++)
        {
            probs[_options.SupportedLanguages[i]] = scores[i];
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

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
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
            Name = _useNativeMode ? "Canary-Native" : "Canary-ONNX",
            Description = $"Canary {_options.Variant} multilingual ASR/ST model (NVIDIA, 2024)",
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        m.AdditionalInfo["Languages"] = string.Join(",", _options.SupportedLanguages);
        return m;
    }





    #endregion

    #region Private Helpers

    private string DecodeTokens(Tensor<T> output)
    {
        var ids = DecodeTokenIds(Engine, output, _options.VocabSize, _options.MaxOutputTokens, _tokenizer.VocabularySize);
        return _tokenizer.Decode(ids.ToList());
    }

    /// <summary>
    /// Token ids from the decoder's vocabulary logits: one row of <paramref name="vocabSize"/> logits per output
    /// position, the id is the row's argmax (NaN never wins, ties go to the lowest id). At most
    /// <paramref name="maxTokens"/> positions are read and ids outside the tokenizer are dropped.
    /// </summary>
    internal static int[] DecodeTokenIds(IEngine engine, Tensor<T> logits, int vocabSize, int maxTokens, int tokenizerSize)
    {
        if (vocabSize < 1) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (logits.Length % vocabSize != 0)
        {
            throw new ArgumentException($"The output holds {logits.Length} values, not whole rows of {vocabSize} vocabulary logits.", nameof(logits));
        }

        int positions = Math.Min(Math.Max(0, maxTokens), logits.Length / vocabSize);
        if (positions == 0) return Array.Empty<int>();

        var rows = engine.TensorSlice(logits.Reshape(new[] { logits.Length / vocabSize, vocabSize }), new[] { 0, 0 }, new[] { positions, vocabSize });
        var cleaned = engine.TensorNanToNum(rows, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity);
        return engine.TensorArgMax(cleaned, 1).ToArray().Where(id => id < tokenizerSize).ToArray();
    }

    private IReadOnlyList<TranscriptionSegment<T>> GenerateTimestamps(Tensor<T> output, string text, double duration)
    {
        var segments = new List<TranscriptionSegment<T>>();
        var words = text.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        int vocabSize = Math.Max(1, _options.VocabSize);
        int positions = output.Length / vocabSize;
        if (words.Length == 0 || positions == 0) return segments;

        // Output positions are spread evenly over the words; a word's confidence is the highest softmax
        // probability its positions gave their decoded token.
        int tokensPerWord = Math.Max(1, positions / words.Length);
        double timePerToken = duration / positions;

        // Word w covers positions [start_w, end_w); positions past the last word belong to no word (-1).
        var starts = new int[words.Length];
        var ends = new int[words.Length];
        var wordOfToken = new int[positions];
        for (int i = 0; i < wordOfToken.Length; i++) wordOfToken[i] = -1;
        int tokenIdx = 0;
        for (int w = 0; w < words.Length; w++)
        {
            starts[w] = tokenIdx;
            ends[w] = Math.Min(positions, tokenIdx + tokensPerWord);
            for (int t = starts[w]; t < ends[w]; t++) wordOfToken[t] = w;
            tokenIdx = ends[w];
        }

        var rows = Engine.TensorSlice(output.Reshape(new[] { output.Length / vocabSize, vocabSize }), new[] { 0, 0 }, new[] { positions, vocabSize });
        var probabilities = Engine.TensorSoftmax(Engine.TensorNanToNum(rows, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity), 1);
        var tokenConfidence = Engine.ReduceMax(probabilities, new[] { 1 }, false);
        var peaks = Engine.ScatterMax(tokenConfidence, new Tensor<int>(wordOfToken, [wordOfToken.Length]), out _, 0, words.Length);
        var confidences = Engine.TensorMin(Engine.TensorMax(peaks, NumOps.Zero), NumOps.One).ToArray();

        for (int w = 0; w < words.Length; w++)
        {
            segments.Add(new TranscriptionSegment<T>
            {
                Text = words[w],
                StartTime = starts[w] * timePerToken,
                EndTime = ends[w] * timePerToken,
                Confidence = confidences[w]
            });
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
            _buffer.AddRange(audioChunk.ToArray());
            int chunkSize = _model._options.SampleRate; // 1 second chunks
            if (_buffer.Count >= chunkSize)
            {
                var frame = new Tensor<T>(_buffer.GetRange(0, chunkSize).ToArray(), [chunkSize]);
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
                var remaining = new Tensor<T>(_buffer.ToArray(), [_buffer.Count]);
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

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
