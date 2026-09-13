using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.Streaming;

/// <summary>
/// Kyutai Moshi: full-duplex spoken dialogue model
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Moshi: a speech-text foundation model for real-time dialogue" (Kyutai, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Moshi is a full-duplex speech-text foundation model that can simultaneously listen and speak, enabling natural spoken dialogue. For ASR, Moshi uses a Mimi neural audio codec to discretize speech into tokens at multiple temporal resolutions. A Tran...</para>
/// <para>
/// Moshi is a full-duplex speech-text foundation model that can simultaneously listen and speak, enabling natural spoken dialogue. For ASR, Moshi uses a Mimi neural audio codec to discretize speech into tokens at multiple temporal resolutions. A Transformer backbone processes both the codec tokens and text tokens in an interleaved fashion. The model can transcribe speech while generating responses, achieving real-time spoken conversation with 200ms turn-taking latency.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Kyutai Moshi model for full-duplex spoken dialogue
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 32000);
/// var model = new KyutaiMoshi&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for real-time dialogue ASR
/// var onnxModel = new KyutaiMoshi&lt;double&gt;(architecture, "moshi.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Moshi: a speech-text foundation model for real-time dialogue", "https://arxiv.org/abs/2410.00037", Year = 2024, Authors = "Kyutai")]
public partial class KyutaiMoshi<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Measured from this model's own output construction, not from a name. <c>PredictCore</c> folds
    /// every layer in <c>Layers</c>, and <c>InitializeLayers</c> fills <c>Layers</c> from
    /// <c>LayerHelper&lt;T&gt;.CreateDefaultConformerLayers(..., vocabSize: _options.VocabSize, ...)</c>,
    /// whose LAST emitted layer is the CTC output head <c>new DenseLayer&lt;T&gt;(vocabSize, identity)</c>.
    /// <c>PostprocessOutput</c> is the identity, so the trailing axis is the CTC vocabulary.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly KyutaiMoshiOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public override ModelOptions GetOptions() => _options;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => true;
    public bool SupportsWordTimestamps => false;

    public KyutaiMoshi(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        KyutaiMoshiOptions? options = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new KyutaiMoshiOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;

        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);

        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { "en" };
        InitializeLayers();
    }

    public KyutaiMoshi(
        NeuralNetworkArchitecture<T> architecture,
        KyutaiMoshiOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new KyutaiMoshiOptions();
        _useNativeMode = true;
        // Paper-faithful fine-tuning recipe from Kyutai's official Moshi code:
        // AdamW with max LR=2e-6, betas=(0.9, 0.95), weight decay=0.1,
        // gradient clipping at 1.0, and a 5%-warmup one-cycle schedule.
        _optimizer = optimizer ?? CreateOneCycleAdamWOptimizer(
            maxLearningRate: _options.LearningRate,
            totalSteps: _options.TotalTrainingSteps,
            pctStart: _options.WarmupFraction,
            weightDecay: _options.WeightDecay,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            maxGradientNorm: _options.MaxGradientNorm);
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { "en" };
        InitializeLayers();
    }

    /// <summary>
    /// Transcribes audio using Moshi's neural codec + Transformer architecture.
    /// Per Kyutai (2024): the Mimi codec discretizes speech into multi-resolution tokens,
    /// and the Transformer backbone processes interleaved speech-text token streams.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> logits;

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            logits = OnnxEncoder.Run(features);
        }
        else
        {
            logits = features;
            foreach (var l in Layers) logits = l.Forward(logits);
        }

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(
        Tensor<T> audio,
        string? language = null,
        bool includeTimestamps = false,
        CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    public string DetectLanguage(Tensor<T> audio)
    {
        var features = PreprocessAudio(audio);
        Tensor<T> logits;

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            logits = OnnxEncoder.Run(features);
        }
        else
        {
            logits = features;
            foreach (var layer in Layers)
                logits = layer.Forward(logits);
        }

        var (tokens, _) = CTCGreedyDecodeWithConfidence(logits);
        return ClassifyLanguageFromTokens(tokens);
    }

    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        var detected = DetectLanguage(audio);
        var result = new Dictionary<string, T>();
        double primaryProbability = 0.85;
        double otherProbability = SupportedLanguages.Count > 1
            ? (1.0 - primaryProbability) / (SupportedLanguages.Count - 1)
            : 0.0;

        foreach (var language in SupportedLanguages)
        {
            result[language] = NumOps.FromDouble(
                language == detected ? primaryProbability : otherProbability);
        }

        return result;
    }

    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new KyutaiMoshiStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultConformerLayers(
                    encoderDim: _options.EncoderDim,
                    numLayers: _options.NumEncoderLayers,
                    numAttentionHeads: _options.NumAttentionHeads,
                    numMels: _options.NumMels,
                    vocabSize: _options.VocabSize,
                    dropoutRate: _options.DropoutRate));
        }
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null)
            return OnnxEncoder.Run(input);

        var current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);

        return current;
    }
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // Pass the model's own non-AMSGrad AdamW explicitly so the
            // fused-Adam fast path engages. The optimizer-null branch falls
            // back to GetOrCreateBaseOptimizer (AMSGrad), which the fused
            // kernel rejects → eager tape path → ~5 s/iter on this Conformer
            // encoder → 120 s test timeout before 30 iters finish.
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
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = _useNativeMode ? "KyutaiMoshi-Native" : "KyutaiMoshi-ONNX",
        Description = "Moshi: full-duplex speech-text dialogue (Kyutai, 2024)",
        FeatureCount = _options.NumMels,
        Complexity = _options.NumEncoderLayers,
        AdditionalInfo = BaseAudioMetadataInfo()
    };

    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits)
    {
        var tokens = new List<int>();
        double totalConfidence = 0;
        int confidenceCount = 0;
        int previousToken = -1;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];

        for (int frame = 0; frame < numFrames && tokens.Count < _options.MaxTextLength; frame++)
        {
            int maxIndex = 0;
            double maxValue = double.NegativeInfinity;
            for (int token = 0; token < vocabSize; token++)
            {
                double value = logits.Rank >= 2
                    ? NumOps.ToDouble(logits[frame, token])
                    : NumOps.ToDouble(logits[token]);
                if (value > maxValue)
                {
                    maxValue = value;
                    maxIndex = token;
                }
            }

            double sumExp = 0;
            for (int token = 0; token < vocabSize; token++)
            {
                double value = logits.Rank >= 2
                    ? NumOps.ToDouble(logits[frame, token])
                    : NumOps.ToDouble(logits[token]);
                sumExp += Math.Exp(value - maxValue);
            }

            double frameConfidence = 1.0 / sumExp;
            if (maxIndex != previousToken && maxIndex > 0)
            {
                tokens.Add(maxIndex);
                totalConfidence += frameConfidence;
                confidenceCount++;
            }

            previousToken = maxIndex;
        }

        return (tokens, confidenceCount > 0 ? totalConfidence / confidenceCount : 0.0);
    }

    private static string TokensToText(List<int> tokens)
    {
        var builder = new System.Text.StringBuilder();
        foreach (var token in tokens)
        {
            if (token > 0 && token <= char.MaxValue)
                builder.Append((char)token);
            else if (token > char.MaxValue && token <= 0x10FFFF)
                builder.Append(char.ConvertFromUtf32(token));
        }

        return builder.ToString().Trim();
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(
        string text,
        double duration,
        double confidence)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<TranscriptionSegment<T>>();

        return new[]
        {
            new TranscriptionSegment<T>
            {
                Text = text,
                StartTime = 0.0,
                EndTime = duration,
                Confidence = NumOps.FromDouble(confidence)
            }
        };
    }

    private string ClassifyLanguageFromTokens(List<int> tokens)
    {
        if (tokens.Count == 0)
            return _options.Language;

        int cjkCount = 0;
        int latinCount = 0;
        foreach (var token in tokens)
        {
            if (token >= 0x4E00 && token <= 0x9FFF)
                cjkCount++;
            else if (token >= 0x41 && token <= 0x7A)
                latinCount++;
        }

        if (cjkCount > latinCount && SupportedLanguages.Contains("zh"))
            return "zh";

        return _options.Language;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(KyutaiMoshi<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        if (disposing)
            OnnxEncoder?.Dispose();

        _disposed = true;
        base.Dispose(disposing);
    }

    private sealed class KyutaiMoshiStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly KyutaiMoshi<T> _model;
        private readonly string _language;
        private readonly List<Tensor<T>> _chunks = new();
        private bool _disposed;

        public KyutaiMoshiStreamingSession(KyutaiMoshi<T> model, string language)
        {
            _model = model;
            _language = language;
        }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(KyutaiMoshiStreamingSession));

            _chunks.Add(audioChunk);
        }

        public TranscriptionResult<T> GetPartialResult()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(KyutaiMoshiStreamingSession));
            if (_chunks.Count == 0)
                return new TranscriptionResult<T> { Language = _language };

            int totalLength = 0;
            foreach (var chunk in _chunks)
                totalLength += chunk.Length;

            var combined = new Tensor<T>(new[] { totalLength });
            int offset = 0;
            foreach (var chunk in _chunks)
            {
                for (int index = 0; index < chunk.Length; index++)
                    combined[offset + index] = chunk[index];
                offset += chunk.Length;
            }

            return _model.Transcribe(combined, _language);
        }

        public TranscriptionResult<T> Finalize()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(KyutaiMoshiStreamingSession));

            var result = GetPartialResult();
            _disposed = true;
            return result;
        }

        public void Dispose()
        {
            _disposed = true;
        }
    }
}
